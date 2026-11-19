/**
 * Normalize the dataset generation timestamp to its UTC calendar date.
 *
 * `generated_at` comes from external JSON, so reject non-string, blank, and
 * unparseable values instead of letting them produce misleading UI metadata.
 */
export function normalizeDatasetGeneratedDate(value) {
    if (typeof value !== 'string')
        return null;
    const timestamp = value.trim();
    if (!timestamp)
        return null;
    const match = /^(\d{4})-(\d{2})-(\d{2})T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$/.exec(timestamp);
    if (!match)
        return null;
    const [, yearText, monthText, dayText] = match;
    const year = Number(yearText);
    const month = Number(monthText);
    const day = Number(dayText);
    const calendarDate = new Date(Date.UTC(year, month - 1, day));
    if (year < 1
        || calendarDate.getUTCFullYear() !== year
        || calendarDate.getUTCMonth() !== month - 1
        || calendarDate.getUTCDate() !== day)
        return null;
    const instant = new Date(timestamp);
    if (!Number.isFinite(instant.getTime()))
        return null;
    return instant.toISOString().slice(0, 10);
}
