from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedFeatureRequest:
    profile: str
    active_features: tuple[str, ...]
    active_blocks: tuple[str, ...]


def _to_string_set(values) -> set[str]:
    if not values:
        return set()
    return {str(value) for value in values if str(value).strip()}


def resolve_feature_request(
    request: dict,
    profile_map: dict[str, list[str] | tuple[str, ...]],
    block_features: dict[str, set[str]],
) -> ResolvedFeatureRequest:
    request = request or {}
    profile = str(request.get("profile", "empty") or "empty")
    available_features = set().union(*block_features.values()) if block_features else set()

    if profile not in profile_map:
        raise ValueError(f"Unknown feature profile: {profile}")

    active_features = _to_string_set(profile_map.get(profile, []))
    include_features = _to_string_set(request.get("include_features"))
    exclude_features = _to_string_set(request.get("exclude_features"))
    exclude_blocks = _to_string_set(request.get("exclude_blocks"))

    unknown_profile_features = sorted(active_features - available_features)
    if unknown_profile_features:
        raise ValueError("Feature profile contains unknown features: " + ", ".join(unknown_profile_features))

    unknown_includes = sorted(include_features - available_features)
    if unknown_includes:
        raise ValueError("FEATURE_BUILD_REQUEST.include_features contains unknown features: " + ", ".join(unknown_includes))

    unknown_excludes = sorted(exclude_features - available_features)
    if unknown_excludes:
        raise ValueError("FEATURE_BUILD_REQUEST.exclude_features contains unknown features: " + ", ".join(unknown_excludes))

    unknown_blocks = sorted(exclude_blocks - set(block_features))
    if unknown_blocks:
        raise ValueError("FEATURE_BUILD_REQUEST.exclude_blocks contains unknown blocks: " + ", ".join(unknown_blocks))

    active_features.update(include_features)
    active_features.difference_update(exclude_features)
    for block_name in exclude_blocks:
        active_features.difference_update(block_features[block_name])

    active_blocks = tuple(
        block_name
        for block_name in sorted(block_features)
        if block_features[block_name].intersection(active_features)
    )
    return ResolvedFeatureRequest(
        profile=profile,
        active_features=tuple(sorted(active_features)),
        active_blocks=active_blocks,
    )
