export const PLATFORM_SORT_KEYS = [
    'score',
    'coverage',
    'num_qubits',
    'provider',
    'device',
    'last_seen',
];
export const DEFAULT_PLATFORM_LIST_ROUTE_STATE = {
    provider: '',
    showRetiredDevices: true,
    sortKey: 'score',
    sortDirection: 'desc',
};
function defaultSortDirection(sortKey) {
    return sortKey === 'device' ? 'asc' : 'desc';
}
export function parsePlatformListRoute(route) {
    const requestedSortKey = String(route.platform_sort || '').trim();
    const hasValidSortKey = PLATFORM_SORT_KEYS.includes(requestedSortKey);
    const sortKey = hasValidSortKey
        ? requestedSortKey
        : DEFAULT_PLATFORM_LIST_ROUTE_STATE.sortKey;
    const requestedSortDirection = String(route.platform_sort_dir || '').trim().toLowerCase();
    const sortDirection = hasValidSortKey && (requestedSortDirection === 'asc' || requestedSortDirection === 'desc')
        ? requestedSortDirection
        : defaultSortDirection(sortKey);
    return {
        provider: String(route.platform_provider || '').trim(),
        showRetiredDevices: String(route.show_retired || '').trim().toLowerCase() !== 'false',
        sortKey,
        sortDirection,
    };
}
export function serializePlatformListRoute(state) {
    const route = {};
    const provider = String(state.provider || '').trim();
    if (provider)
        route.platform_provider = provider;
    if (!state.showRetiredDevices)
        route.show_retired = 'false';
    if (state.sortKey !== DEFAULT_PLATFORM_LIST_ROUTE_STATE.sortKey
        || state.sortDirection !== DEFAULT_PLATFORM_LIST_ROUTE_STATE.sortDirection) {
        route.platform_sort = state.sortKey;
        route.platform_sort_dir = state.sortDirection;
    }
    return route;
}
