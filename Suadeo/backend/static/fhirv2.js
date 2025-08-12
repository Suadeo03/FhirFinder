function displayV2FhirResult(result, dataset) {
    console.log('displayV2FhirResult called with:', result);
    
    if (!result?.results?.[0]) {
        console.error('Invalid V2 FHIR result structure:', result);
        return '<div class="error">Invalid response structure</div>';
    }
    
    const firstResult = result.results[0];
    
    
    const resource = firstResult.resource || 'Unknown Resource';
    const localId = firstResult.local_id || 'No Local ID';
    const fhirDetail = firstResult.fhir_detail || 'No FHIR details available';
    const fhirVersion = firstResult.fhir_version || 'Unknown';
    const hl7v2Field = firstResult.hl7v2_field || 'No HL7 V2 field';
    const hl7v2FieldDetail = firstResult.hl7v2_field_detail || 'No HL7 V2 field details available';
    const hl7v2Version = firstResult.hl7v2_field_version || 'Unknown';
    const subDetail = firstResult.sub_detail || '';
    const similarityScore = firstResult.similarity_score || 0;

    return `<div class="success">
        
        <h2>V2-FHIR Mapping Recommendation</h2>
        <h3>Resource: ${escapeHtml(resource)}</h3>
        <div class="mapping-header">
            <span class="versions"><strong>FHIR:</strong> ${escapeHtml(fhirVersion)} | <strong>HL7 V2:</strong> ${escapeHtml(hl7v2Version)}</span>
        </div>
        
        ${subDetail ? `<div class="sub-detail"><strong>Full FHIR Path: </strong>${escapeHtml(resource)}.${escapeHtml(subDetail)}</div>` : ''}
        
        <details open>
            <summary>FHIR Details</summary>
            <div class="fhir-detail">${escapeHtml(fhirDetail)}</div>
        </details>
        
        <details open>
            <summary>HL7 V2 Mapping</summary>
            <div class="v2-mapping">
                <div><strong>Field:</strong> ${escapeHtml(hl7v2Field)}</div>
                <div class="v2-detail">${escapeHtml(hl7v2FieldDetail)}</div>
            </div>
        </details>
        
        <div class="mapping-metadata">
            <div class="similarity-score">
                <small>Similarity Score: ${(similarityScore * 100).toFixed(1)}%</small>
            </div>
            <div class="dataset-info">
                <small>Dataset ID: ${firstResult.dataset_id || 'Unknown'}</small>
            </div>
        </div>
    </div>`;
}
window.displayV2FhirResult = displayV2FhirResult;