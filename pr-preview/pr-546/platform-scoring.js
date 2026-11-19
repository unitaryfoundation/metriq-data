function finiteNumber(value) {
    if (typeof value !== 'number' && (typeof value !== 'string' || !value.trim()))
        return null;
    const result = Number(value);
    return Number.isFinite(result) ? result : null;
}
function groupKey(component) {
    return JSON.stringify([component.group, component.aggregation]);
}
function readInputs(components) {
    if (!components || typeof components !== 'object' || Array.isArray(components))
        return null;
    const items = [];
    const groups = new Map();
    let denominator = 0;
    for (const [name, value] of Object.entries(components)) {
        if (!value || typeof value !== 'object')
            return null;
        const weight = finiteNumber(value.weight);
        if (weight === null || weight < 0)
            return null;
        const group = value.group == null ? null : value.group;
        if (group !== null && typeof group !== 'string')
            return null;
        const groupWeight = group === null ? 1 : finiteNumber(value.group_weight);
        const subWeight = group === null ? weight : finiteNumber(value.sub_weight);
        const aggregation = group === null ? 'arithmetic' : value.aggregation;
        if (groupWeight === null || groupWeight < 0 || subWeight === null || subWeight < 0
            || (aggregation !== 'arithmetic' && aggregation !== 'harmonic'))
            return null;
        const raw = finiteNumber(value.raw);
        // Earlier payloads contain raw measurements without their normalization
        // baseline. Do not silently replace raw-first scoring with a sum of ratios.
        if (group !== null && aggregation === 'arithmetic' && raw !== null
            && (!Object.prototype.hasOwnProperty.call(value, 'baseline')
                || (value.direction !== 'higher' && value.direction !== 'lower')))
            return null;
        const item = {
            name, weight, group, groupWeight, subWeight, aggregation,
            normalized: finiteNumber(value.normalized),
            raw,
            baseline: finiteNumber(value.baseline),
            direction: value.direction === 'lower' ? 'lower' : 'higher',
        };
        items.push(item);
        denominator += weight;
        if (group !== null) {
            const key = groupKey(item);
            let entry = groups.get(key);
            if (!entry) {
                entry = { weight: groupWeight, aggregation, items: [] };
                groups.set(key, entry);
            }
            if (entry.weight !== groupWeight)
                return null;
            entry.items.push(item);
        }
    }
    return denominator > 0 && Number.isFinite(denominator) ? { items, groups, denominator } : null;
}
function rawSubscore(items) {
    const usable = items.filter((item) => item.subWeight > 0
        && item.raw !== null && item.baseline !== null);
    if (!usable.length)
        return null;
    let device = 0;
    let baseline = 0;
    let present = 0;
    for (const item of usable) {
        device += item.subWeight * item.raw;
        baseline += item.subWeight * item.baseline;
        present += item.subWeight;
    }
    const lower = usable.some((item) => item.direction === 'lower');
    if ((lower ? device : baseline) <= 0)
        return null;
    return { value: 100 * (lower ? baseline / device : device / baseline), present };
}
function evaluate(inputs) {
    let numerator = 0;
    for (const item of inputs.items) {
        if (item.group === null && item.normalized !== null) {
            numerator += item.weight * item.normalized;
        }
    }
    // Match the canonical evaluator's arithmetic-before-harmonic addition order.
    const groups = Array.from(inputs.groups.values()).sort((left, right) => Number(left.aggregation === 'harmonic') - Number(right.aggregation === 'harmonic'));
    for (const group of groups) {
        const total = group.items.reduce((sum, item) => sum + Math.max(0, item.subWeight), 0);
        if (group.aggregation === 'harmonic') {
            const present = group.items.filter((item) => item.subWeight > 0 && item.normalized !== null);
            if (total <= 0 || !present.length || present.some((item) => item.normalized <= 0))
                continue;
            const presentWeight = present.reduce((sum, item) => sum + item.subWeight, 0);
            const inverseSum = present.reduce((sum, item) => sum + item.subWeight / item.normalized, 0);
            if (inverseSum > 0)
                numerator += group.weight * (presentWeight / inverseSum) * (presentWeight / total);
            continue;
        }
        const subscore = total > 0 ? rawSubscore(group.items) : null;
        if (subscore) {
            numerator += group.weight * subscore.value * (subscore.present / total);
        }
        else {
            // Derived metrics can contain normalized values without raw measurements.
            for (const item of group.items) {
                if (item.normalized !== null)
                    numerator += item.weight * item.normalized;
            }
        }
    }
    const score = numerator / inputs.denominator;
    return Number.isFinite(score) ? score : null;
}
export function calculateMetriqScore(components) {
    const inputs = readInputs(components);
    return inputs ? evaluate(inputs) : null;
}
function scoringModes(inputs) {
    return new Map(inputs.items.map((item) => [
        item.name,
        item.group !== null && item.aggregation === 'arithmetic'
            && rawSubscore(inputs.groups.get(groupKey(item)).items) ? 'raw' : 'normalized',
    ]));
}
/** Apply the same missing-measurement mask to both devices, retaining all weights. */
export function calculateOverlapScores(leftComponents, rightComponents) {
    const left = readInputs(leftComponents);
    const right = readInputs(rightComponents);
    if (!left || !right)
        return { left: null, right: null, sharedNames: [] };
    const leftModes = scoringModes(left);
    const rightModes = scoringModes(right);
    const rightByName = new Map(right.items.map((item) => [item.name, item]));
    const sharedNames = [];
    const sharedNormalized = new Set();
    for (const item of left.items) {
        const other = rightByName.get(item.name);
        const mode = leftModes.get(item.name);
        if (!other || mode !== rightModes.get(item.name))
            continue;
        const hasNormalized = item.normalized !== null && other.normalized !== null;
        const shared = mode === 'raw'
            ? item.subWeight > 0 && other.subWeight > 0
                && item.raw !== null && item.baseline !== null
                && other.raw !== null && other.baseline !== null
            : hasNormalized;
        if (!shared)
            continue;
        sharedNames.push(item.name);
        if (hasNormalized)
            sharedNormalized.add(item.name);
    }
    sharedNames.sort();
    const shared = new Set(sharedNames);
    for (const inputs of [left, right]) {
        for (const item of inputs.items) {
            if (!shared.has(item.name)) {
                item.raw = null;
                item.baseline = null;
                item.normalized = null;
            }
            else {
                if (!sharedNormalized.has(item.name))
                    item.normalized = null;
                // Both sides must use normalized fallback together. Retained raw values
                // must not make one side switch aggregation method after masking.
                if (leftModes.get(item.name) === 'normalized') {
                    item.raw = null;
                    item.baseline = null;
                }
            }
        }
    }
    return { left: evaluate(left), right: evaluate(right), sharedNames };
}
