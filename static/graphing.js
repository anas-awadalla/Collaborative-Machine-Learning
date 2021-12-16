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
    };
}

let updateChart = () => {};
updateChartFactory("#time-accuracy", timeAccuracySpec).then((u) => (updateChart = u));