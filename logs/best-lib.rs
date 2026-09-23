//! The Rust translation of `reference/version.py`.

use std::cmp::Ordering;

/// A parsed semantic version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub prerelease: Option<String>,
    pub build: Option<String>,
}

fn is_number(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b >= b'0' && b <= b'9')
}

fn validate_part(s: &str) -> Result<(), String> {
    if s.is_empty() {
        return Err("empty identifier".to_string());
    }
    if s.len() > 1 && s.starts_with('0') && is_number(s) {
        return Err("leading zero in numeric identifier".to_string());
    }
    Ok(())
}

pub fn parse(s: &str) -> Result<Version, String> {
    if s.is_empty() {
        return Err("empty string".to_string());
    }

    // Split build (+)
    let (rest, build) = match s.find('+') {
        Some(idx) => {
            let (r, b) = s.split_at(idx);
            let b = &b[1..];
            if b.is_empty() {
                return Err("empty build metadata".to_string());
            }
            // validate build parts
            for part in b.split('.') {
                if part.is_empty() {
                    return Err("empty build identifier".to_string());
                }
                // build parts can have leading zeros, alphanumeric + hyphen
                for b_byte in part.bytes() {
                    if !((b_byte >= b'0' && b_byte <= b'9')
                        || (b_byte >= b'a' && b_byte <= b'z')
                        || (b_byte >= b'A' && b_byte <= b'Z')
                        || b_byte == b'-')
                    {
                        return Err("invalid character in build".to_string());
                    }
                }
            }
            (r, Some(b.to_string()))
        }
        None => (s, None),
    };

    // Split prerelease (-)
    let (rest, prerelease) = match rest.find('-') {
        Some(idx) => {
            let (r, p) = rest.split_at(idx);
            let p = &p[1..];
            if p.is_empty() {
                return Err("empty prerelease".to_string());
            }
            for part in p.split('.') {
                if part.is_empty() {
                    return Err("empty prerelease identifier".to_string());
                }
                for b_byte in part.bytes() {
                    if !((b_byte >= b'0' && b_byte <= b'9')
                        || (b_byte >= b'a' && b_byte <= b'z')
                        || (b_byte >= b'A' && b_byte <= b'Z')
                        || b_byte == b'-')
                    {
                        return Err("invalid character in prerelease".to_string());
                    }
                }
                validate_part(part)?;
            }
            (r, Some(p.to_string()))
        }
        None => (rest, None),
    };

    // Core version X.Y.Z
    let parts: Vec<&str> = rest.split('.').collect();
    if parts.len() != 3 {
        return Err(format!("expected 3 core version parts, got {}", parts.len()));
    }

    let parse_num = |s: &str| -> Result<u64, String> {
        if s.is_empty() {
            return Err("empty version number".to_string());
        }
        if s.len() > 1 && s.starts_with('0') {
            return Err("leading zero in version number".to_string());
        }
        for b in s.bytes() {
            if b < b'0' || b > b'9' {
                return Err("non-numeric character in version number".to_string());
            }
        }
        s.parse::<u64>().map_err(|e| e.to_string())
    };

    let major = parse_num(parts[0])?;
    let minor = parse_num(parts[1])?;
    let patch = parse_num(parts[2])?;

    Ok(Version {
        major,
        minor,
        patch,
        prerelease,
        build,
    })
}

pub fn to_string(v: &Version) -> String {
    let mut out = format!("{}.{}.{}", v.major, v.minor, v.patch);
    if let Some(ref p) = v.prerelease {
        out.push('-');
        out.push_str(p);
    }
    if let Some(ref b) = v.build {
        out.push('+');
        out.push_str(b);
    }
    out
}

fn compare_prerelease(a: &Option<String>, b: &Option<String>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (Some(_), None) => Ordering::Less, // Having prerelease is lower than not having
        (None, Some(_)) => Ordering::Greater,
        (Some(sa), Some(sb)) => {
            let parts_a: Vec<&str> = sa.split('.').collect();
            let parts_b: Vec<&str> = sb.split('.').collect();

            for (pa, pb) in parts_a.iter().zip(parts_b.iter()) {
                let num_a = pa.parse::<u64>();
                let num_b = pb.parse::<u64>();

                match (num_a, num_b) {
                    (Ok(na), Ok(nb)) => {
                        let ord = na.cmp(&nb);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                    }
                    (Ok(_), Err(_)) => return Ordering::Less, // Numeric is lower than non-numeric
                    (Err(_), Ok(_)) => return Ordering::Greater,
                    (Err(_), Err(_)) => {
                        let ord = pa.cmp(pb);
                        if ord != Ordering::Equal {
                            return ord;
                        }
                    }
                }
            }
            parts_a.len().cmp(&parts_b.len())
        }
    }
}

pub fn compare(a: &Version, b: &Version) -> Ordering {
    let ord = a.major.cmp(&b.major);
    if ord != Ordering::Equal {
        return ord;
    }
    let ord = a.minor.cmp(&b.minor);
    if ord != Ordering::Equal {
        return ord;
    }
    let ord = a.patch.cmp(&b.patch);
    if ord != Ordering::Equal {
        return ord;
    }
    compare_prerelease(&a.prerelease, &b.prerelease)
}

pub fn bump_major(v: &Version) -> Version {
    Version {
        major: v.major + 1,
        minor: 0,
        patch: 0,
        prerelease: None,
        build: None,
    }
}

pub fn bump_minor(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor + 1,
        patch: 0,
        prerelease: None,
        build: None,
    }
}

pub fn bump_patch(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor,
        patch: v.patch + 1,
        prerelease: None,
        build: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid() {
        let v = parse("1.2.3-alpha.1+build.123").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(v.prerelease.as_deref(), Some("alpha.1"));
        assert_eq!(v.build.as_deref(), Some("build.123"));
        assert_eq!(to_string(&v), "1.2.3-alpha.1+build.123");
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse("01.2.3").is_err());
        assert!(parse("1.2").is_err());
        assert!(parse("1.2.3-01").is_err());
        assert!(parse("1.2.3+").is_err());
    }

    #[test]
    fn test_compare() {
        let v1 = parse("1.0.0-alpha").unwrap();
        let v2 = parse("1.0.0-alpha.1").unwrap();
        let v3 = parse("1.0.0").unwrap();
        assert_eq!(compare(&v1, &v2), Ordering::Less);
        assert_eq!(compare(&v2, &v3), Ordering::Less);
        assert_eq!(compare(&v3, &v1), Ordering::Greater);
    }

    #[test]
    fn test_bumps() {
        let v = parse("1.2.3-alpha+build").unwrap();
        assert_eq!(to_string(&bump_major(&v)), "2.0.0");
        assert_eq!(to_string(&bump_minor(&v)), "1.3.0");
        assert_eq!(to_string(&bump_patch(&v)), "1.2.4");
    }
}
