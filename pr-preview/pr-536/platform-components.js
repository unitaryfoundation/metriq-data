const platformComponentCollator = new Intl.Collator('en', {
    numeric: true,
    sensitivity: 'base',
});
function platformComponentGroup([name, component]) {
    const group = typeof component?.group === 'string' ? component.group.trim() : '';
    return group || name;
}
export function sortPlatformScoreComponents(entries) {
    return entries.slice().sort((a, b) => {
        const groupDiff = platformComponentCollator.compare(platformComponentGroup(a), platformComponentGroup(b));
        if (groupDiff !== 0)
            return groupDiff;
        return platformComponentCollator.compare(a[0], b[0]);
    });
}
function commandArgument(value) {
    if (typeof value !== 'string')
        return null;
    const trimmed = value.trim();
    if (!trimmed || /[\u0000-\u001f\u007f]/.test(trimmed))
        return null;
    return trimmed;
}
function objectRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
        ? value
        : null;
}
function suiteLookupKey(value) {
    const normalized = commandArgument(value);
    return normalized?.toLocaleLowerCase('en-US') ?? null;
}
function suiteBenchmarkSelector(benchmark) {
    const rawComponent = benchmark.component;
    if (rawComponent !== undefined && rawComponent !== null) {
        if (typeof rawComponent !== 'string')
            return null;
        if (/[\u0000-\u001f\u007f]/.test(rawComponent))
            return null;
        const component = rawComponent.trim();
        if (component)
            return component;
    }
    return commandArgument(benchmark.name);
}
export function resolveMetriqGymSuiteDispatch(suiteDefinition, group) {
    const definition = objectRecord(suiteDefinition);
    const requestedAlias = suiteLookupKey(group);
    if (!definition || !requestedAlias)
        return null;
    const suite = commandArgument(definition.name);
    const benchmarks = definition.benchmarks;
    if (!suite || !Array.isArray(benchmarks))
        return null;
    let component = null;
    let componentKey = null;
    for (const value of benchmarks) {
        const benchmark = objectRecord(value);
        const config = objectRecord(benchmark?.config);
        if (!benchmark || !config)
            continue;
        if (suiteLookupKey(config.benchmark_name) !== requestedAlias)
            continue;
        const selector = suiteBenchmarkSelector(benchmark);
        if (!selector)
            return null;
        const selectorKey = selector.toLocaleLowerCase('en-US');
        if (componentKey !== null && selectorKey !== componentKey)
            return null;
        if (component === null)
            component = selector;
        componentKey = selectorKey;
    }
    return component ? { suite, component } : null;
}
function quotePosixShellArgument(value) {
    if (/^[A-Za-z0-9_./:@%+=,-]+$/.test(value))
        return value;
    return `'${value.replace(/'/g, `'"'"'`)}'`;
}
export function buildMetriqGymDispatchInstructions({ provider, device, suite, component, runtimeDeviceId, }) {
    const providerArgument = commandArgument(provider);
    const platformDevice = commandArgument(device);
    const suiteArgument = commandArgument(suite);
    const suiteComponent = commandArgument(component);
    if (!providerArgument || !platformDevice || !suiteArgument || !suiteComponent)
        return null;
    const isAws = ['aws', 'braket'].includes(providerArgument.toLocaleLowerCase('en-US'));
    // Runtime identifiers are optional metadata. Ignore unusable values rather
    // than suppressing otherwise valid dispatch guidance.
    const suppliedRuntimeDevice = commandArgument(runtimeDeviceId);
    const validAwsRuntimeDevice = isAws && suppliedRuntimeDevice && /^arn:[^:]+:braket:[^:]+:[^:]*:device\/.+$/i.test(suppliedRuntimeDevice)
        ? suppliedRuntimeDevice
        : null;
    const requiresRuntimeDeviceId = isAws && !validAwsRuntimeDevice;
    const deviceArgument = validAwsRuntimeDevice
        ?? (requiresRuntimeDeviceId ? `<full Braket ARN for ${platformDevice}>` : platformDevice);
    const command = [
        `mgym suite dispatch ${quotePosixShellArgument(suiteArgument)} \\`,
        `  --component ${quotePosixShellArgument(suiteComponent)} \\`,
        `  --provider ${quotePosixShellArgument(providerArgument)} \\`,
        `  --device ${quotePosixShellArgument(deviceArgument)}`,
    ].join('\n');
    return { command, suite: suiteArgument, suiteComponent, requiresRuntimeDeviceId };
}
