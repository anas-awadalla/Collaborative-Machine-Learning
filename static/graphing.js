const BG = "#1a1e25";
const FG = "#abb2bf";

const timeAccuracySpec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    mark: { type: "line", point: true },
    data: { name: "table" },
    encoding: {
        x: { field: "time", type: "quantitative", title: "Time(s)" },
        y: {
            field: "accuracy",
            type: "quantitative",
            scale: { domain: [0, 100] },
            axis: { tickCount: 5 },
            title: "Accuracy (%)",
        },
    },
    config: {
        background: BG,
        point: { fill: "#FF7E70" },
        line: { stroke: FG },
        axis: { titleColor: FG, labelColor: FG, gridColor: "#444" },
        title: { fill: FG },
    },
    title: "Model accuracy over time",
};

const timeSpentSpec = {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    mark: { type: "area" },
    data: { name: "table" },
    encoding: {
        x: {
            field: "i",
            type: "quantitative",
            title: "Step",
            axis: { tickMinStep: 1 },
        },
        y: {
            field: "value",
            type: "quantitative",
            aggregate: "mean",
            title: "Time (s)",
            axis: { tickMinStep: 2 },
        },
        color: { field: "task", type: "nominal" },
    },
    title: "Time spent on computation vs network",
    config: {
        background: BG,
        line: { stroke: FG },
        axis: { titleColor: FG, labelColor: FG, gridColor: "#444" },
        legend: { labelColor: FG, titleColor: FG },
        title: { fill: FG },
    },
};

async function updateChartFactory(element, vlSpec) {
    const res = await vegaEmbed(element, vlSpec, {
        renderer: "svg",
        actions: false,
    });

    let oldData = [];
    return (fullData) => {
        newData = fullData.slice(oldData.length);
        const changeSet = vega.changeset().insert(newData);
        oldData = fullData;
        res.view.change("table", changeSet).run();
        res.view.resize();
    };
}

let updateChart = () => {};
Promise.all([
    updateChartFactory("#time-accuracy", timeAccuracySpec),
    updateChartFactory("#time-spent", timeSpentSpec),
]).then(([timeAccuracyUpdater, timeSpentUpdater]) => {
    updateChart = (graphData) => {
        console.log('gd', graphData);
        timeAccuracyUpdater(
            graphData.time_accuracy.slice(1).map(([t, a]) => ({
                time: t,
                accuracy: a,
            }))
        );
        if (graphData.time_log[UUID]) {
            timeSpentUpdater(
                graphData.time_log[UUID].map(([ct, nt], i) => [
                    { task: "network", value: nt / 1000, i },
                    { task: "computation", value: ct / 1000, i },
                ]).flat()
            );
        }
    };
});