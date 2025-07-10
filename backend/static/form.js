function displayCodeResult(result, dataset) {

    let fhirObjects = result.results[0].fhir_resource[0];
    let textSummary = fhirObjects.text && fhirObjects.text.div ? fhirObjects.text.div : 'No text summary available';
    fhirObjects.text.div = 'Text context removed for brevity';
    const uniqueId = `json-${Date.now()}`;

    return `<div class="success">
    <h2>Best match from ${dataset} dataset: ${JSON.stringify(result.results[0].resource_type)}</h2>
    ${JSON.stringify(textSummary)}<br/>
    <details>
    <summary>Description</summary>
    ${JSON.stringify(result.results[0].description)}
    </details>
    <details>
    <summary>JSON</summary>
    <button onclick="copyJSONtoClipboard('${uniqueId}', this)" class="copy-button-mini">📋</button>
    <pre><code id="${uniqueId}">${JSON.stringify(fhirObjects, null, 2)}</code></pre>
    </details>
    <details>
    <summary>Constraints</summary>
    ${JSON.stringify(result.results[0].must_have[0])}<br/>
    ${JSON.stringify(result.results[0].must_support[0])}<br/>
    ${JSON.stringify(result.results[0].invariants[0])}<br/>
    </details>
    <details>
    <summary>Specification URL</summary>
    <a href=${JSON.stringify(result.results[0].resource_url)}>${JSON.stringify(result.results[0].resource_url)}</a>
    </details>
    <details>
    <summary>Match Metrics [Development ONLY View]</summary>
    ${JSON.stringify(result.results[0].match_reasons[0])}<br/>
    ${JSON.stringify(result.results[0].match_reasons[1])}
    </details>
    </div>`;

}