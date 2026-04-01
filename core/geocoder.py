"""GPS 좌표 → 지역명 변환"""
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=2000)
def get_location_name(lat: float, lon: float) -> Optional[str]:
    """GPS 좌표를 읽기 좋은 지역명으로 변환"""
    try:
        import reverse_geocoder as rg
        results = rg.search([(lat, lon)], verbose=False)
        if not results:
            return None

        r = results[0]
        name = r.get("name", "").strip()
        country = r.get("cc", "")
        admin1 = r.get("admin1", "").strip()

        if not name:
            name = admin1
        if not name:
            return None

        # 나라명 매핑 (ISO 코드 → 표시명)
        country_names = {
            "KR": "South Korea",
            "US": "USA", "JP": "Japan", "CN": "China", "DE": "Germany",
            "FR": "France", "GB": "UK", "IT": "Italy", "ES": "Spain",
            "AU": "Australia", "TH": "Thailand", "VN": "Vietnam",
            "CH": "Switzerland", "AT": "Austria", "NZ": "New Zealand",
            "TW": "Taiwan", "HK": "Hong Kong", "SG": "Singapore",
            "PH": "Philippines", "ID": "Indonesia", "MY": "Malaysia",
            "IN": "India", "TR": "Turkey", "GR": "Greece", "PT": "Portugal",
            "NL": "Netherlands", "BE": "Belgium", "SE": "Sweden",
            "NO": "Norway", "DK": "Denmark", "FI": "Finland",
            "PL": "Poland", "CZ": "Czech Republic", "HU": "Hungary",
            "CA": "Canada", "MX": "Mexico", "BR": "Brazil",
        }
        country_display = country_names.get(country, country)
        return f"{name}, {country_display}"

    except Exception:
        return None


def coords_to_str(gps: list) -> Optional[str]:
    """[lat, lon] 리스트를 지역명으로 변환"""
    if not gps or len(gps) < 2:
        return None
    return get_location_name(float(gps[0]), float(gps[1]))
