function displayFormResult(result, dataset) {

    let formObjects = result;



    return `<div class="success">
    <h2>Best domain match from ${dataset} dataset: ${result.results[0].domain}</h2>
    <h3>Screening Tool:</h3><p>${result.results[0].screening_tool}</p>
    <h3>Question:</h3> <p>${result.results[0].question}</p>
    <h3>Query:</h3><p>${result.query}</p>
    <h3>Answers:</h3><p>${result.results[0].answer_concept}</p>
    <details>
    <summary>LOINC</summary>
    </details>
    <details>
    <summary>FHIR Structure</summary>
    </details>
    </div>`;

}

window.displayFormResult = displayFormResult;