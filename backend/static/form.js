function displayFormResult(result, dataset) {

    let formObjects = result;

    let fhirResource = {
        resourceType: "QuestionnaireResponse",
        id: "example",
        meta: {
            profile: [
                "http://hl7.org/fhir/us/core/StructureDefinition/us-core-questionnaireresponse|8.0.0"
            ]
        },
        text: {
            status: "generated",
            div: "html omitted for brevity"
        },
        authored: new Date().toISOString(),
        item: []
    };



    return `<div class="success">
    <h2>Best domain match from ${dataset} dataset: ${result.results[0].domain}</h2>
    <h3>Screening Tool:</h3><p>${result.results[0].screening_tool}</p>
    <h3>Question:</h3> <p>${result.results[0].question}</p>
    <h3>Query:</h3><p>${result.query}</p>
    <h3>Answers:</h3><p>${result.results[0].answer_concept}</p>
    <details>
    <summary>LOINC</summary>
    <p>LOINC Code: ${result.results[0].loinc_question_code} - Name: ${result.results[0].loinc_question_name_long}</p>
    <p>LOINC Answer: ${result.results[0].loinc_answer} - Concept: ${result.results[0].loinc_concept}</p>
    <p>Panel: ${result.results[0].loinc_panel_code} - Panel Name: ${result.results[0].loinc_panel_name}</p>
    </details>
    <details>
    <summary>FHIR Structure</summary>
    ${JSON.stringify(fhirResource,null,2)}
    </details>
    </div>`;

}

window.displayFormResult = displayFormResult;