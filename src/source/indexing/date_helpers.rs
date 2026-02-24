use chrono::NaiveDate;

/// Parse a date from a folder name formatted as `MM-DD-YYYY` or `YYYY-MM-DD`.
///
/// Accepts either `-` or `_` separators. Returns `None` when parsing fails.
pub fn parse_publication_date_from_folder(folder: &str) -> Option<NaiveDate> {
    let normalized = folder.replace('_', "-");
    if let Ok(date) = NaiveDate::parse_from_str(&normalized, "%m-%d-%Y") {
        return Some(date);
    }
    if let Ok(date) = NaiveDate::parse_from_str(&normalized, "%Y-%m-%d") {
        return Some(date);
    }
    None
}

/// Parse a date from a year folder plus a date folder like `MM-DD-YYYY` or `MM-DD`.
///
/// Examples: `2026/02-14-2026`, `2026/02-14`, or `2026/02_14_2026`. The folder
/// may include suffixes after `--` (e.g. `04-14-2025--INCOMPLETE`). Returns `None`
/// when no valid date is found.
pub fn parse_publication_date_from_year_folder(year_str: &str, folder: &str) -> Option<NaiveDate> {
    let year = year_str.parse::<i32>().ok()?;
    let core = folder.split("--").next().unwrap_or(folder).trim();
    let normalized = core.replace('_', "-");
    if let Ok(date) = NaiveDate::parse_from_str(&normalized, "%m-%d-%Y") {
        return Some(date);
    }
    if let Ok(date) = NaiveDate::parse_from_str(&normalized, "%Y-%m-%d") {
        return Some(date);
    }
    if let Some((month, day)) = parse_month_day(&normalized) {
        return NaiveDate::from_ymd_opt(year, month, day);
    }
    None
}

/// Parse a month-range folder like `Jan + Feb 2026` using the year from `year_str`.
///
/// Uses the last month token encountered in the folder name and anchors the date
/// to day 1 (e.g. `Dec 2025 + Jan 2026` yields January 2026). Returns `None` if
/// the year or month cannot be parsed.
pub fn parse_publication_date_from_month_range_folder(
    year_str: &str,
    folder: &str,
) -> Option<NaiveDate> {
    let year = year_str.parse::<i32>().ok()?;
    let month = parse_month_from_range_folder(folder, year)?;
    NaiveDate::from_ymd_opt(year, month, 1)
}

/// Parse a `MM-DD` string into (month, day) with basic bounds checks.
fn parse_month_day(value: &str) -> Option<(u32, u32)> {
    let mut parts = value.split('-');
    let month = parts.next()?.parse::<u32>().ok()?;
    let day = parts.next()?.parse::<u32>().ok()?;
    if (1..=12).contains(&month) && (1..=31).contains(&day) {
        Some((month, day))
    } else {
        None
    }
}

/// Extract the last month token from a month-range folder name.
fn parse_month_from_range_folder(folder: &str, _target_year: i32) -> Option<u32> {
    let tokens: Vec<String> = folder
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect();

    tokens
        .iter()
        .filter_map(|token| month_token_to_number(token))
        .next_back()
}

/// Convert a lowercase month token to a month number (1-12).
fn month_token_to_number(token: &str) -> Option<u32> {
    match token {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "sept" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_publication_date_from_folder_formats() {
        assert_eq!(
            parse_publication_date_from_folder("02-14-2026"),
            NaiveDate::from_ymd_opt(2026, 2, 14)
        );
        assert_eq!(
            parse_publication_date_from_folder("2026-02-14"),
            NaiveDate::from_ymd_opt(2026, 2, 14)
        );
        assert_eq!(
            parse_publication_date_from_folder("02_14_2026"),
            NaiveDate::from_ymd_opt(2026, 2, 14)
        );
        assert_eq!(parse_publication_date_from_folder(" 02-14-2026 "), None);
        assert_eq!(parse_publication_date_from_folder("13-01-2026"), None);
        assert_eq!(parse_publication_date_from_folder("02-32-2026"), None);
        assert_eq!(parse_publication_date_from_folder("2026-13-01"), None);
        assert_eq!(parse_publication_date_from_folder("not-a-date"), None);
    }

    #[test]
    fn parses_publication_date_from_year_folder_formats() {
        assert_eq!(
            parse_publication_date_from_year_folder("2026", "02-14-2026"),
            NaiveDate::from_ymd_opt(2026, 2, 14)
        );
        assert_eq!(
            parse_publication_date_from_year_folder("2026", "02-14"),
            NaiveDate::from_ymd_opt(2026, 2, 14)
        );
        assert_eq!(
            parse_publication_date_from_year_folder("2026", "02_14_2026"),
            NaiveDate::from_ymd_opt(2026, 2, 14)
        );
        assert_eq!(
            parse_publication_date_from_year_folder("2026", "04-14-2025--INCOMPLETE"),
            NaiveDate::from_ymd_opt(2025, 4, 14)
        );
        assert_eq!(
            parse_publication_date_from_year_folder("2026", " 02-14 "),
            NaiveDate::from_ymd_opt(2026, 2, 14)
        );
        assert_eq!(
            parse_publication_date_from_year_folder("2026", "13-14"),
            None
        );
        assert_eq!(
            parse_publication_date_from_year_folder("2026", "02-32"),
            None
        );
        assert_eq!(
            parse_publication_date_from_year_folder("20xx", "02-14"),
            None
        );
        assert_eq!(parse_publication_date_from_year_folder("2026", "bad"), None);
    }

    #[test]
    fn parses_publication_date_from_month_range_folder() {
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "Jan + Feb 2026"),
            NaiveDate::from_ymd_opt(2026, 2, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "Feb 2026"),
            NaiveDate::from_ymd_opt(2026, 2, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "March + April"),
            NaiveDate::from_ymd_opt(2026, 4, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "SEPT 2026"),
            NaiveDate::from_ymd_opt(2026, 9, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "(Oct) 2026"),
            NaiveDate::from_ymd_opt(2026, 10, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "Nov 2026"),
            NaiveDate::from_ymd_opt(2026, 11, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "Nov. 2026"),
            NaiveDate::from_ymd_opt(2026, 11, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2025", "Nov. + Dec. 2025"),
            NaiveDate::from_ymd_opt(2025, 12, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "2026 Feb"),
            NaiveDate::from_ymd_opt(2026, 2, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "Dec 2025 + Jan 2026"),
            NaiveDate::from_ymd_opt(2026, 1, 1)
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("20xx", "Feb 2026"),
            None
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", ""),
            None
        );
        assert_eq!(
            parse_publication_date_from_month_range_folder("2026", "2026"),
            None
        );
    }
}
