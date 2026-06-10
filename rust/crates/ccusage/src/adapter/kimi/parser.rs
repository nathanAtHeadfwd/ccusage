use std::{
    fs,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use jiff::tz::TimeZone as JiffTimeZone;
use serde::Deserialize;

use super::super::jsonl;
use crate::{
    LoadedEntry, PricingMap, Result, TimestampMs, TokenUsageRaw, UsageEntry, UsageMessage,
    apply_total_token_fallback, calculate_cost_for_usage, cli::CostMode, fast::LinePrefilter,
    format_date_tz, missing_pricing_model_for_candidates,
};

const DEFAULT_MODEL: &str = "kimi-for-coding";
const DEFAULT_PROVIDER: &str = "moonshot";
const KIMI_FOR_CODING_K2_6_CUTOFF_MS: i64 = 1_776_698_890_072;

/// Kimi `config.json` document, used to read the configured display model.
#[derive(Debug, Deserialize)]
struct KimiConfig {
    #[serde(default, deserialize_with = "jsonl::non_empty_string")]
    model: Option<String>,
}

/// A single Kimi wire JSONL line in the legacy `StatusUpdate` format.
#[derive(Debug, Deserialize)]
struct KimiStatusUpdateLine {
    #[serde(default, deserialize_with = "jsonl::non_empty_string")]
    r#type: Option<String>,
    message: Option<KimiStatusUpdateMessage>,
    #[serde(default, deserialize_with = "jsonl::lenient_f64")]
    timestamp: Option<f64>,
}

/// The `message` block carried by a legacy Kimi wire line.
#[derive(Debug, Deserialize)]
struct KimiStatusUpdateMessage {
    #[serde(default, deserialize_with = "jsonl::non_empty_string")]
    r#type: Option<String>,
    payload: Option<KimiStatusUpdatePayload>,
}

/// The `message.payload` block carrying token usage and the message id.
#[derive(Debug, Deserialize)]
struct KimiStatusUpdatePayload {
    token_usage: Option<KimiStatusUpdateTokenUsage>,
    #[serde(default, deserialize_with = "jsonl::non_empty_string")]
    message_id: Option<String>,
}

/// Token counts reported under `message.payload.token_usage`.
#[derive(Debug, Default, Deserialize)]
struct KimiStatusUpdateTokenUsage {
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    input_other: u64,
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    output: u64,
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    input_cache_creation: u64,
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    input_cache_read: u64,
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    total: u64,
}

/// A single Kimi Code CLI wire JSONL line in the `usage.record` format.
#[derive(Debug, Deserialize)]
struct KimiUsageRecordLine {
    #[serde(default, deserialize_with = "jsonl::non_empty_string")]
    r#type: Option<String>,
    #[serde(default, deserialize_with = "jsonl::non_empty_string")]
    model: Option<String>,
    usage: Option<KimiUsageRecordTokenUsage>,
    #[serde(default, deserialize_with = "jsonl::lenient_i64")]
    time: Option<i64>,
}

/// Token counts reported under the `usage` field of a `usage.record` line.
#[derive(Debug, Default, Deserialize)]
#[allow(non_snake_case)]
struct KimiUsageRecordTokenUsage {
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    inputOther: u64,
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    output: u64,
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    inputCacheCreation: u64,
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    inputCacheRead: u64,
    #[serde(default, deserialize_with = "jsonl::lenient_u64")]
    total: u64,
}

#[derive(Debug, Clone)]
pub(super) struct KimiUsageEntry {
    timestamp: TimestampMs,
    timestamp_text: String,
    session_id: String,
    model: String,
    message_id: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
    cache_creation_tokens: u64,
    cache_read_tokens: u64,
    extra_total_tokens: u64,
}

pub(super) fn read_wire_file(path: &Path) -> Result<Vec<KimiUsageEntry>> {
    let fallback_model = read_model_from_config(path);
    let fallback_timestamp = file_modified_timestamp(path);
    let content = fs::read(path)?;

    let status_prefilter = LinePrefilter::all(&[br#""StatusUpdate""#, br#""token_usage""#]);
    let mut entries: Vec<KimiUsageEntry> =
        jsonl::records::<KimiStatusUpdateLine>(&content, Some(&status_prefilter))
            .filter_map(|line| {
                status_update_line_to_entry(&line, path, &fallback_model, fallback_timestamp)
            })
            .collect();

    let usage_record_prefilter = LinePrefilter::all(&[br#""usage.record""#]);
    entries.extend(
        jsonl::records::<KimiUsageRecordLine>(&content, Some(&usage_record_prefilter))
            .filter_map(|line| usage_record_line_to_entry(&line, path, fallback_timestamp)),
    );

    entries.sort_by_key(|entry| entry.timestamp);
    Ok(entries)
}

fn read_model_from_config(file_path: &Path) -> String {
    let Some(root) = kimi_root_from_wire_path(file_path) else {
        return DEFAULT_MODEL.to_string();
    };
    let Ok(content) = fs::read_to_string(root.join("config.json")) else {
        return DEFAULT_MODEL.to_string();
    };
    let Ok(config) = serde_json::from_str::<KimiConfig>(&content) else {
        return DEFAULT_MODEL.to_string();
    };
    config.model.unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

fn kimi_root_from_wire_path(file_path: &Path) -> Option<PathBuf> {
    file_path
        .parent()?
        .parent()?
        .parent()?
        .parent()
        .map(Path::to_path_buf)
}

fn file_modified_timestamp(path: &Path) -> TimestampMs {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
        .map(TimestampMs::from_millis)
        .unwrap_or(TimestampMs::UNIX_EPOCH)
}

fn status_update_line_to_entry(
    line: &KimiStatusUpdateLine,
    file_path: &Path,
    model: &str,
    fallback_timestamp: TimestampMs,
) -> Option<KimiUsageEntry> {
    if line.r#type.as_deref() == Some("metadata") {
        return None;
    }
    let message = line.message.as_ref()?;
    if message.r#type.as_deref() != Some("StatusUpdate") {
        return None;
    }
    let payload = message.payload.as_ref()?;
    let token_usage = payload.token_usage.as_ref()?;
    let usage = TokenUsageRaw {
        input_tokens: token_usage.input_other,
        output_tokens: token_usage.output,
        cache_creation_input_tokens: token_usage.input_cache_creation,
        cache_read_input_tokens: token_usage.input_cache_read,
        speed: None,
        cache_creation: None,
    };
    let (usage, extra_total_tokens) = apply_total_token_fallback(usage, 0, token_usage.total);
    if crate::total_usage_tokens(usage) + extra_total_tokens == 0 {
        return None;
    }
    let timestamp = line
        .timestamp
        .and_then(timestamp_from_seconds)
        .unwrap_or(fallback_timestamp);
    Some(build_kimi_entry(
        usage,
        extra_total_tokens,
        timestamp,
        model.to_string(),
        extract_session_id(file_path),
        payload.message_id.clone(),
    ))
}

fn usage_record_line_to_entry(
    line: &KimiUsageRecordLine,
    file_path: &Path,
    fallback_timestamp: TimestampMs,
) -> Option<KimiUsageEntry> {
    if line.r#type.as_deref() != Some("usage.record") {
        return None;
    }
    let usage_value = line.usage.as_ref()?;
    let usage = TokenUsageRaw {
        input_tokens: usage_value.inputOther,
        output_tokens: usage_value.output,
        cache_creation_input_tokens: usage_value.inputCacheCreation,
        cache_read_input_tokens: usage_value.inputCacheRead,
        speed: None,
        cache_creation: None,
    };
    let (usage, extra_total_tokens) = apply_total_token_fallback(usage, 0, usage_value.total);
    if crate::total_usage_tokens(usage) + extra_total_tokens == 0 {
        return None;
    }
    let timestamp = line
        .time
        .map(TimestampMs::from_millis)
        .unwrap_or(fallback_timestamp);
    let model = line
        .model
        .as_deref()
        .map(normalize_kimi_model)
        .unwrap_or_else(|| read_model_from_config(file_path));
    Some(build_kimi_entry(
        usage,
        extra_total_tokens,
        timestamp,
        model,
        extract_session_id(file_path),
        None,
    ))
}

fn build_kimi_entry(
    usage: TokenUsageRaw,
    extra_total_tokens: u64,
    timestamp: TimestampMs,
    model: String,
    session_id: String,
    message_id: Option<String>,
) -> KimiUsageEntry {
    KimiUsageEntry {
        timestamp,
        timestamp_text: crate::format_rfc3339_millis(timestamp),
        session_id,
        model,
        message_id,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_creation_tokens: usage.cache_creation_input_tokens,
        cache_read_tokens: usage.cache_read_input_tokens,
        extra_total_tokens,
    }
}

fn timestamp_from_seconds(seconds: f64) -> Option<TimestampMs> {
    if !seconds.is_finite() {
        return None;
    }
    let millis = (seconds * 1000.0).trunc();
    if millis < i64::MIN as f64 || millis > i64::MAX as f64 {
        return None;
    }
    Some(TimestampMs::from_millis(millis as i64))
}

fn extract_session_id(file_path: &Path) -> String {
    let components: Vec<_> = file_path.components().collect();
    for (i, component) in components.iter().enumerate() {
        if let Component::Normal(name) = component {
            if name.to_str() == Some("sessions") && i + 2 < components.len() {
                // Old format: sessions/group/session/wire.jsonl
                // New format: sessions/workspace/session/agents/main/wire.jsonl
                if let Component::Normal(session) = &components[i + 2] {
                    return session.to_str().unwrap_or("unknown").to_string();
                }
            }
        }
    }
    file_path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("unknown")
        .to_string()
}

fn normalize_kimi_model(model: &str) -> String {
    // Strip provider prefixes like "kimi-code/" to get the base model name
    if let Some(pos) = model.rfind('/') {
        model[pos + 1..].to_string()
    } else {
        model.to_string()
    }
}

pub(super) fn kimi_entry_key(entry: &KimiUsageEntry) -> String {
    format!(
        "{}:{}:{}:{}:{}:{}:{}:{}:{}",
        entry.session_id,
        entry.message_id.as_deref().unwrap_or_default(),
        entry.timestamp_text,
        entry.model,
        entry.input_tokens,
        entry.output_tokens,
        entry.cache_creation_tokens,
        entry.cache_read_tokens,
        entry.extra_total_tokens
    )
}

pub(super) fn kimi_entry_to_loaded(
    entry: KimiUsageEntry,
    tz: Option<&JiffTimeZone>,
    mode: CostMode,
    pricing: &PricingMap,
) -> LoadedEntry {
    let usage = TokenUsageRaw {
        input_tokens: entry.input_tokens,
        output_tokens: entry.output_tokens,
        cache_creation_input_tokens: entry.cache_creation_tokens,
        cache_read_input_tokens: entry.cache_read_tokens,
        speed: None,
        cache_creation: None,
    };
    let cost = calculate_kimi_cost(&entry, mode, pricing, usage);
    let missing_pricing_model = missing_kimi_pricing(&entry, mode, pricing, usage);
    let data = UsageEntry {
        session_id: Some(entry.session_id.clone()),
        timestamp: entry.timestamp_text,
        version: None,
        message: UsageMessage {
            usage,
            model: Some(entry.model.clone()),
            id: entry.message_id.clone(),
        },
        cost_usd: None,
        request_id: None,
        is_api_error_message: None,
        is_sidechain: None,
    };
    LoadedEntry {
        date: format_date_tz(entry.timestamp, tz),
        timestamp: entry.timestamp,
        project: Arc::from("kimi"),
        session_id: Arc::from(entry.session_id),
        project_path: Arc::from("Kimi"),
        cost,
        extra_total_tokens: entry.extra_total_tokens,
        credits: None,
        message_count: None,
        model: Some(entry.model),
        usage_limit_reset_time: None,
        missing_pricing_model,
        data,
    }
}

fn calculate_kimi_cost(
    entry: &KimiUsageEntry,
    mode: CostMode,
    pricing: &PricingMap,
    usage: TokenUsageRaw,
) -> f64 {
    match mode {
        CostMode::Display => 0.0,
        CostMode::Auto | CostMode::Calculate => {
            for candidate in model_candidates(entry) {
                if pricing.find(&candidate).is_some() {
                    return calculate_cost_for_usage(
                        Some(&candidate),
                        usage,
                        None,
                        CostMode::Calculate,
                        Some(pricing),
                    );
                }
            }
            0.0
        }
    }
}

fn missing_kimi_pricing(
    entry: &KimiUsageEntry,
    mode: CostMode,
    pricing: &PricingMap,
    usage: TokenUsageRaw,
) -> Option<String> {
    if mode == CostMode::Display {
        return None;
    }
    missing_pricing_model_for_candidates(
        &entry.model,
        model_candidates(entry),
        crate::total_usage_tokens(usage).saturating_add(entry.extra_total_tokens),
        Some(pricing),
    )
}

fn model_candidates(entry: &KimiUsageEntry) -> Vec<String> {
    let mut candidates = Vec::new();
    if entry.model == DEFAULT_MODEL {
        candidates.push(kimi_for_coding_pricing_model(entry.timestamp).to_string());
    }
    candidates.extend([
        format!("{DEFAULT_PROVIDER}/{}", entry.model),
        format!("kimi/{}", entry.model),
        entry.model.clone(),
    ]);
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|candidate| seen.insert(candidate.clone()));
    candidates
}

fn kimi_for_coding_pricing_model(timestamp: TimestampMs) -> &'static str {
    if timestamp.as_millis() < KIMI_FOR_CODING_K2_6_CUTOFF_MS {
        "moonshot/kimi-k2.5"
    } else {
        "moonshot/kimi-k2.6"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccusage_test_support::fs_fixture;

    #[test]
    fn falls_back_to_total_tokens_when_kimi_parts_are_missing() {
        let fixture = fs_fixture!({
            "sessions/group/session-a/wire.jsonl": "",
        });
        let file = fixture.path("sessions/group/session-a/wire.jsonl");
        let line = serde_json::from_value::<KimiStatusUpdateLine>(serde_json::json!({
            "timestamp": 1770983427.123,
            "message": {
                "type": "StatusUpdate",
                "payload": {
                    "token_usage": {
                        "total": 432
                    }
                }
            }
        }))
        .unwrap();

        let entry =
            status_update_line_to_entry(&line, &file, "kimi-k2", TimestampMs::UNIX_EPOCH).unwrap();

        assert_eq!(entry.output_tokens, 432);
        assert_eq!(entry.extra_total_tokens, 0);
    }

    #[test]
    fn prices_default_kimi_model_by_timestamp_without_changing_display_model() {
        let pricing = PricingMap::load_embedded();
        let usage = TokenUsageRaw {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 20,
            cache_read_input_tokens: 10,
            speed: None,
            cache_creation: None,
        };
        let before_cutoff = KimiUsageEntry {
            timestamp: TimestampMs::from_millis(1_776_698_890_071),
            timestamp_text: "2026-04-20T15:28:10.071Z".to_string(),
            session_id: "session-a".to_string(),
            model: DEFAULT_MODEL.to_string(),
            message_id: Some("msg-before".to_string()),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_creation_tokens: usage.cache_creation_input_tokens,
            cache_read_tokens: usage.cache_read_input_tokens,
            extra_total_tokens: 0,
        };
        let at_cutoff = KimiUsageEntry {
            timestamp: TimestampMs::from_millis(1_776_698_890_072),
            timestamp_text: "2026-04-20T15:28:10.072Z".to_string(),
            message_id: Some("msg-at".to_string()),
            ..before_cutoff.clone()
        };

        let before_cost = calculate_kimi_cost(&before_cutoff, CostMode::Calculate, &pricing, usage);
        let at_cost = calculate_kimi_cost(&at_cutoff, CostMode::Calculate, &pricing, usage);
        let loaded = kimi_entry_to_loaded(at_cutoff, None, CostMode::Calculate, &pricing);

        assert!((before_cost - 0.000226).abs() < f64::EPSILON);
        assert!((at_cost - 0.00032035).abs() < f64::EPSILON);
        assert_eq!(loaded.model.as_deref(), Some(DEFAULT_MODEL));
        assert_eq!(loaded.data.message.model.as_deref(), Some(DEFAULT_MODEL));
    }

    #[test]
    fn parses_kimi_code_usage_record_format() {
        let fixture = fs_fixture!({
            "sessions/workspace/session-a/agents/main/wire.jsonl": "",
        });
        let file = fixture.path("sessions/workspace/session-a/agents/main/wire.jsonl");
        let line = serde_json::from_value::<KimiUsageRecordLine>(serde_json::json!({
            "type": "usage.record",
            "model": "kimi-code/kimi-for-coding",
            "usage": {
                "inputOther": 100,
                "output": 50,
                "inputCacheRead": 10,
                "inputCacheCreation": 20,
            },
            "usageScope": "turn",
            "time": 1780925702183_i64,
        }))
        .unwrap();

        let entry = usage_record_line_to_entry(&line, &file, TimestampMs::UNIX_EPOCH).unwrap();

        assert_eq!(entry.session_id, "session-a");
        assert_eq!(entry.model, "kimi-for-coding");
        assert_eq!(entry.input_tokens, 100);
        assert_eq!(entry.output_tokens, 50);
        assert_eq!(entry.cache_read_tokens, 10);
        assert_eq!(entry.cache_creation_tokens, 20);
        assert_eq!(entry.timestamp.as_millis(), 1_780_925_702_183);
    }

    #[test]
    fn skips_kimi_code_usage_record_with_zero_tokens() {
        let fixture = fs_fixture!({
            "sessions/workspace/session-a/agents/main/wire.jsonl": "",
        });
        let file = fixture.path("sessions/workspace/session-a/agents/main/wire.jsonl");
        let line = serde_json::from_value::<KimiUsageRecordLine>(serde_json::json!({
            "type": "usage.record",
            "model": "kimi-code/kimi-for-coding",
            "usage": {
                "inputOther": 0,
                "output": 0,
                "inputCacheRead": 0,
                "inputCacheCreation": 0,
            },
            "usageScope": "turn",
            "time": 1780925702183_i64,
        }))
        .unwrap();

        assert!(usage_record_line_to_entry(&line, &file, TimestampMs::UNIX_EPOCH).is_none());
    }

    #[test]
    fn extract_session_id_from_old_format() {
        let fixture = fs_fixture!({
            "sessions/group/session-old/wire.jsonl": "",
        });
        let file = fixture.path("sessions/group/session-old/wire.jsonl");
        assert_eq!(extract_session_id(&file), "session-old");
    }

    #[test]
    fn extract_session_id_from_kimi_code_format() {
        let fixture = fs_fixture!({
            "sessions/workspace/session-new/agents/main/wire.jsonl": "",
        });
        let file = fixture.path("sessions/workspace/session-new/agents/main/wire.jsonl");
        assert_eq!(extract_session_id(&file), "session-new");
    }

    #[test]
    fn normalize_kimi_model_strips_provider_prefix() {
        assert_eq!(
            normalize_kimi_model("kimi-code/kimi-for-coding"),
            "kimi-for-coding"
        );
        assert_eq!(normalize_kimi_model("kimi/kimi-k2"), "kimi-k2");
        assert_eq!(normalize_kimi_model("kimi-for-coding"), "kimi-for-coding");
        assert_eq!(normalize_kimi_model(""), "");
    }

    #[test]
    fn read_wire_file_parses_both_formats() {
        let fixture = fs_fixture!({
            "sessions/workspace/session-new/agents/main/wire.jsonl": [
                r#"{"type":"metadata","protocol_version":"1.3"}"#,
                r#"{"type":"usage.record","model":"kimi-code/kimi-for-coding","usage":{"inputOther":100,"output":50,"inputCacheRead":10,"inputCacheCreation":20},"usageScope":"turn","time":1780925702183}"#,
                r#"{"timestamp":1770983427.123,"message":{"type":"StatusUpdate","payload":{"token_usage":{"input_other":11,"output":21,"input_cache_read":31,"input_cache_creation":41},"message_id":"msg-1"}}}"#,
            ]
            .join("\n"),
        });
        let entries =
            read_wire_file(&fixture.path("sessions/workspace/session-new/agents/main/wire.jsonl"))
                .unwrap();

        assert_eq!(entries.len(), 2);
        // StatusUpdate has the earlier timestamp (2026-02-13), usage.record is later (2026-06-08).
        assert_eq!(entries[0].input_tokens, 11);
        assert_eq!(entries[0].model, DEFAULT_MODEL);
        assert_eq!(entries[0].message_id.as_deref(), Some("msg-1"));
        assert_eq!(entries[1].input_tokens, 100);
        assert_eq!(entries[1].model, "kimi-for-coding");
    }
}
