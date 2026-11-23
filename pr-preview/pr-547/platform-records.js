import { calculateMetriqScore } from './platform-scoring.js';
function finiteNumber(value) {
    if (typeof value !== 'number' && typeof value !== 'string')
        return null;
    if (typeof value === 'string' && !value.trim())
        return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}
function rawDirection(run, metric) {
    // The data pipeline uses higher-is-better when no direction is specified.
    const direction = String(run?.rawDirections?.[metric] ?? 'higher').toLowerCase();
    return direction === 'higher' || direction === 'lower' ? direction : null;
}
/** Replace latest components with their best matching records, then score canonically.
 *
 * Matching uses the published record's timestamp to identify its instance; records
 * of a different width or circuit configuration must never replace that instance.
 * Raw values and normalization inputs move together so arithmetic groups can still
 * aggregate measurements before normalization in both full and overlap scores.
 */
export function adjustMetriqScoreForRecords(detail, runsIndex) {
    const score = detail?.metriq_score;
    const components = score?.components;
    if (!components || typeof components !== 'object' || Array.isArray(components)
        || !runsIndex?.size || typeof runsIndex.get !== 'function')
        return detail;
    // Older data lacks the inputs required to reproduce the published aggregation.
    // Keep its published score instead of reverting to a sum of individual ratios.
    if (calculateMetriqScore(components) === null)
        return detail;
    const provider = String(detail?.provider || '');
    const device = String(detail?.device || '');
    const rawGroups = new Set(Object.values(components).filter((component) => (component?.group != null && component?.aggregation !== 'harmonic'
        && finiteNumber(component?.raw) !== null)).map((component) => component.group));
    const adjustedComponents = { ...components };
    let changed = false;
    for (const [name, component] of Object.entries(components)) {
        const requiresRaw = component?.aggregation !== 'harmonic' && rawGroups.has(component?.group);
        const anchorTs = requiresRaw
            ? component?.raw_timestamp ?? component?.normalized_timestamp ?? component?.timestamp
            : component?.normalized_timestamp ?? component?.timestamp ?? component?.raw_timestamp;
        if (!anchorTs)
            continue;
        const entries = runsIndex.get(`${provider}::${device}::${String(component?.group || '')}`);
        if (!Array.isArray(entries))
            continue;
        const anchors = entries.filter((entry) => entry?.run?.timestamp === anchorTs);
        const signatures = new Set(anchors.map((entry) => entry.sig));
        // A timestamp shared by different instances cannot identify the component
        // reliably; choosing the first match could substitute a different width.
        if (signatures.size !== 1)
            continue;
        const anchor = anchors[0];
        if (!anchor || typeof anchor.sig !== 'string')
            continue;
        const metric = String(component?.metric || '');
        const selfBaseline = component?.baseline_is_self === true;
        let bestNormalized = finiteNumber(component?.normalized);
        let bestRaw = finiteNumber(component?.raw);
        let selected = null;
        for (const entry of entries) {
            if (entry?.sig !== anchor.sig || !entry?.run?.timestamp)
                continue;
            const run = entry.run;
            const raw = finiteNumber(run?.rawResults?.[metric]);
            const baseline = selfBaseline ? raw : finiteNumber(run?.normalizationBaselines?.[metric]);
            const direction = rawDirection(run, metric);
            const runNormalized = finiteNumber(run?.normalizedScores?.[metric]);
            const normalized = selfBaseline && runNormalized !== null ? 100 : runNormalized;
            if (requiresRaw && (raw === null || baseline === null || direction === null))
                continue;
            if (normalized === null && !(selfBaseline && requiresRaw && raw !== null))
                continue;
            // A baseline device is normalized against itself in platform composites.
            // An older run's ratio against today's baseline cannot increase that 100.
            const improves = selfBaseline
                ? raw !== null && (bestRaw === null || (direction === 'lower' ? raw < bestRaw : raw > bestRaw))
                : normalized !== null && (bestNormalized === null || normalized > bestNormalized);
            if (!improves)
                continue;
            bestNormalized = normalized;
            bestRaw = raw;
            selected = {
                ...component,
                normalized,
                normalized_available: normalized !== null,
                timestamp: normalized !== null ? run.timestamp : null,
                normalized_timestamp: normalized !== null ? run.timestamp : null,
                raw,
                raw_available: raw !== null,
                raw_timestamp: raw !== null ? run.timestamp : null,
                baseline,
                direction: direction ?? 'higher',
            };
        }
        if (selected) {
            adjustedComponents[name] = selected;
            changed = true;
        }
    }
    if (!changed)
        return detail;
    const value = calculateMetriqScore(adjustedComponents);
    if (value === null)
        return detail;
    // These optional display values describe the published latest records. The
    // calculator derives groups from their inputs, so do not retain stale values.
    for (const [name, component] of Object.entries(adjustedComponents)) {
        const { group_subscore: _publishedSubscore, ...adjusted } = component;
        adjustedComponents[name] = adjusted;
    }
    return { ...detail, metriq_score: { ...score, value, components: adjustedComponents } };
}
