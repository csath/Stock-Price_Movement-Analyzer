const GOOG = "GOOG"
const AAPL = "AAPL"
const IBM = "IBM"
const NOK = "NOK"
const AMZN = "AMZN"

function formatData(list) {
    for (var j = 0; j < list.length; j++) {
        list[j].Day_1 = parseFloat(list[j].Day_1);
        list[j].Day_2 = parseFloat(list[j].Day_2);
        list[j].Day_3 = parseFloat(list[j].Day_3);
        list[j].Day_4 = parseFloat(list[j].Day_4);
        list[j].Day_5 = parseFloat(list[j].Day_5);
        list[j].Actual = parseFloat(list[j].Actual);
        list[j].Date = AmCharts.stringToDate(list[j].Date, "YYYY-MM-DD");
    }
    list.splice(-1);
    return list;
}

function loadData(ticker) {
    var loc = 'PredictedData/';
    var temp = ticker.toUpperCase();
    if (temp == GOOG) {
        loc += GOOG;
    }
    else if (temp == IBM) {
        loc += IBM;
    }
    else if (temp == AMZN) {
        loc += AMZN;
    }
    else if (temp == NOK) {
        loc += NOK;
    }
    else if (temp == AAPL) {
        loc += AAPL;
    }

    var chartData1 = [];



    AmCharts.loadFile(loc + "/final.csv", {}, function (data) {
        var chartData = AmCharts.parseCSV(data, {
            "useColumnNames": true
        });
        chartData1 = formatData(chartData);
        // console.log(chartData1);

        var chart = AmCharts.makeChart("chartdiv", {
            "type": "stock",
            "theme": "light",
            "dataSets": [{
                "fieldMappings": [{
                    "fromField": "Actual",
                    "toField": "Actual"
                }, {
                    "fromField": "Day_1",
                    "toField": "Day_1"
                }],
                "dataProvider": chartData1,
                "categoryField": "Date"
            }],

            "panels": [{
                "showCategoryAxis": true,
                "title": "Stock Price",

                "stockGraphs": [{
                    "id": "g1",
                    "title": "Graph #1",
                    "lineThickness": 2,
                    "valueField": "Actual",
                    "useDataSetColors": false
                }, {
                    "id": "g2",
                    "title": "Graph #2",
                    "lineThickness": 2,
                    "valueField": "Day_1",
                    "useDataSetColors": false
                }],
                "stockLegend": {
                    "periodValueTextComparing": "[[percents.Actual.close]]%",
                    "periodValueTextRegular": "[[Actual.A]]"
                }
            }],

            "chartScrollbarSettings": {
                "graph": "g1"
            },

            "chartCursorSettings": {
                "valueBalloonsEnabled": true,
                "fullWidth": true,
                "cursorAlpha": 0.1,
                "valueLineBalloonEnabled": true,
                "valueLineEnabled": true,
                "valueLineAlpha": 0.5
            },

            "periodSelector": {
                "position": "left",
                "periods": [{
                    "period": "MM",
                    "selected": true,
                    "count": 1,
                    "label": "1 month"
                }, {
                    "period": "YYYY",
                    "count": 1,
                    "label": "1 year"
                }, {
                    "period": "YTD",
                    "label": "YTD"
                }, {
                    "period": "MAX",
                    "label": "MAX"
                }]
            },

            "dataSetSelector": {
                "position": "left"
            },

            /*"export": {
            //  "enabled": true
            }*/
        });
    });
}



loadData(AAPL);

