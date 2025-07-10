// Tab switching functionality
let currentTab = 'resources';

function switchTab(tabName) {
 
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
    currentTab = tabName;
    clearResult();
}

// Get the current query value based on active tab
function getCurrentQuery() {
    switch(currentTab) {
        case 'resources':
            return document.getElementById('resourceQuery').value.trim();
        case 'search':
            return document.getElementById('searchQuery').value.trim();
        case 'form':
            return document.getElementById('formQuery').value.trim();
        case 'mapping':
            return document.getElementById('mappingQuery').value.trim();
        default:
            return '';
    }
}

class FhirTextQuery {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async getQuery(query, dataset = 'fhir') {
        try {
       
            let endpoint = '/api/v1/search/';
            switch(dataset) {
                case 'terminology':
                    endpoint = '/api/v1/parameters/';
                    break;
                case 'assessments':
                    endpoint = '/api/v1/form/';
                    break;
                case 'mapping':
                    endpoint = '/api/v1/mapping/';
                    break;
            }

            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                mode: 'cors',
                body: JSON.stringify({
                    query: `${query}`,
                    limit: 1,
                    dataset: dataset
                })
            });

            console.log(`Response status: ${response.status}`);

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`API Error: ${response.status} - ${errorText}`);
                throw new Error(`API Error: ${response.status} - ${errorText}`);
            }

            const data = await response.json();
            console.log('Successful response:', data);
            return data;

        } catch (error) {
            console.error('Error loading code:', error);
            throw error;
        }
    }
}

let newQuery = new FhirTextQuery();

async function lookupBtn() {
    const btn = document.getElementById('lookupBtn'); 
    const text = getCurrentQuery();

    if (!text) {
        showResult('<div class="error">Please enter something to search</div>');
        return;
    }

    try {
        btn.disabled = true;
        showResult(`<div class="loading">Searching ${currentTab} dataset...</div>`);
        
        const result = await newQuery.getQuery(text, currentTab);

        function hasResults(result) {
            if (!result) { return false; }
            if ((result.results && result.results.length > 0) || (result.data && result.data.length > 0)) {
                return true;
            } 
            return false;
        }

        if (hasResults(result)) {
            showResult(displayCodeResult(result, currentTab));
        } else {
            showResult(`<div class="error">No results found in ${currentTab} dataset</div>`);
        }
    } catch (error) {
        console.error('Frontend error:', error);
        showResult(`<div class="error">Error searching ${currentTab}: ${error.message}</div>`);
    } finally {
        btn.disabled = false; 
    }
}

function showResult(html) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
}



function displayCodeResult(result, dataset) {
    console.log('displayCodeResult called with:', result);
    
    if (!result?.results?.[0]) {
        console.error('Invalid result structure:', result);
        return '<div class="error">Invalid response structure</div>';
    }
    
    const firstResult = result.results[0];
    console.log('First result:', firstResult);
    
    let fhirObjects = {};
    let textSummary = 'No text summary available';
    
    try {

        if (firstResult.fhir_resource && 
            Array.isArray(firstResult.fhir_resource) && 
            firstResult.fhir_resource.length > 0) {
            
            const originalFhirObject = firstResult.fhir_resource[0];
            
            if (originalFhirObject?.text?.div && originalFhirObject.text.div !== 'Text context removed for brevity') {
                textSummary = originalFhirObject.text.div;
            }
            
            fhirObjects = JSON.parse(JSON.stringify(originalFhirObject));
            
            if (fhirObjects.text) {
                fhirObjects.text.div = 'Text context removed for brevity';
            }
        } else {
            console.warn('Not a valid array:', firstResult.fhir_resource);
            fhirObjects = { error: 'No FHIR resource data available' };
        }
    } catch (error) {
        console.error('Error processing:', error);
        fhirObjects = { error: 'Error processing FHIR resource data' };
    }
    
    const uniqueId = `json-${Date.now()}`;
    
    const resourceType = firstResult.resource_type || 'Unknown';
    const description = firstResult.description || 'No description available';
    
  
    const mustHave = Array.isArray(firstResult.must_have) && firstResult.must_have.length > 0 
        ? firstResult.must_have[0] 
        : 'None specified';
        
    const mustSupport = Array.isArray(firstResult.must_support) && firstResult.must_support.length > 0 
        ? firstResult.must_support[0] 
        : 'None specified';
        
    const invariants = Array.isArray(firstResult.invariants) && firstResult.invariants.length > 0 
        ? firstResult.invariants[0] 
        : 'None specified';
    
    const resourceUrl = firstResult.resource_url && firstResult.resource_url !== 'None available' 
        ? firstResult.resource_url 
        : '#';

    return `<div class="success">
        <h2>Best match from ${dataset} dataset: ${escapeHtml(resourceType)}</h2>
        <div class="text-summary">${textSummary}</div><br/>
        
        <details>
            <summary>Description</summary>
            <div class="description">${escapeHtml(description)}</div>
        </details>
        
        <details>
            <summary>JSON Resource</summary>
            <button onclick="copyJSONtoClipboard('${uniqueId}', this)" class="copy-button-mini">Copy JSON</button>
            <pre><code id="${uniqueId}">${JSON.stringify(fhirObjects, null, 2)}</code></pre>
        </details>
        
        <details>
            <summary>Constraints</summary>
            <div><strong>Must Have:</strong> ${escapeHtml(mustHave)}</div>
            <div><strong>Must Support:</strong> ${escapeHtml(mustSupport)}</div>
            <div><strong>Invariants:</strong> ${escapeHtml(invariants)}</div>
        </details>
        
        <details>
            <summary>Specification</summary>
            ${resourceUrl !== '#' 
                ? `<a href="${escapeHtml(resourceUrl)}" target="_blank" rel="noopener">${escapeHtml(resourceUrl)}</a>`
                : 'No specification URL available'
            }
        </details>
        
        <div class="similarity-score">
            <small>Similarity Score: ${(firstResult.similarity_score * 100).toFixed(1)}%</small>
        </div>
    </div>`;
}

// Helper function to escape HTML and prevent XSS
function escapeHtml(text) {
    if (typeof text !== 'string') {
        return String(text);
    }
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
function copyJSONtoClipboard(elementId) {
    const element = document.getElementById(elementId);
    const text = element.textContent;
    
    navigator.clipboard.writeText(text).then(() => {

    }).catch(err => {
        console.error('Failed to copy: ', err);
    });
}


function clearResult() {
    const ids = ['result', 'resourceQuery', 'searchQuery', 'formQuery', 'mappingQuery'];
    
    ids.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = '';
            element.value = '';
            if (id === 'result') {
                element.style.display = 'none';
            }
        }
    });
}



function testRawUMLS() {
    showResult('<div class="loading">🔄 Adding to use case...</div>');
}

function loadCode() {
    showResult('<div class="loading">🔄 Loading code...</div>');
}

function getCode() {
    showResult('<div class="loading">🔄 Getting code...</div>');
}




document.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        const activeInput = document.querySelector('.tab-content.active input');
        if (activeInput === document.activeElement) {
            lookupBtn();
        }
    }
});

