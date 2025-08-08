
// Tab switching functionality
let currentTab = 'resources';
let currentResult = null;
function switchTab(tabName, event) {
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
                case 'form':
                    endpoint = '/api/v1/forms/search';
                    break;
                case 'mapping':
                    endpoint = '/api/v1/mapping/';
                    break;
                default:
                    endpoint = '/api/v1/search/'
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
        currentResult = result;

        function hasResults(result) {
            if (!result) { return false; }
            if ((result.results && result.results.length > 0) || (result.data && result.data.length > 0)) {
                return true;
            } 
            return false;
        }

        if (hasResults(result)) {
            if (currentTab === 'resources') {
                showResult(displayCodeResult(result, currentTab));
            }
            else if (currentTab === 'form') {
                showResult(displayFormResult(result, currentTab));
            } 
        } else {
            showResult(`<div class="error">No results found in ${currentTab} dataset</div>`);
            currentResult = null;
        }
    } catch (error) {
        console.error('Frontend error:', error);
        showResult(`<div class="error">Error searching ${currentTab}: ${error.message}</div>`);
        currentResult = null;
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

class FeedbackService {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    //case for each feedback type tab
    async sendFeedback(query, id, feedbackType, original_score, contextInfo) {
        try {
      
            const response = await fetch(`${this.baseUrl}/api/v1/feedback/record`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                mode: 'cors',
                body: JSON.stringify({
                    query: query,
                    profile_id: id,
                    feedback_type: feedbackType,
                    session_id: 'default-session',
                    user_id: 'default-user',
                    original_score: original_score || 0.0,
                    context_info: contextInfo || {},
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`API Error: ${response.status} - ${errorText}`);
                throw new Error(`API Error: ${response.status} - ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error sending feedback:', error);
            throw error;
        }
    }
}

function positiveResponse() {
    sendFeedbackResponse('positive');
}

function negativeResponse() {
    sendFeedbackResponse('negative');
}


async function sendFeedbackResponse(feedbackType) {
    if (!currentResult) {
        showResult('<div class="error">No search result available for feedback.</div>');
        return;
    }
    //modify based on current tab
    const query = getCurrentQuery();
    let id, originalScore, context_info;
    const feedbackService = new FeedbackService();
    if (currentTab === 'resources') {
        id = currentResult.results[0].id;
        originalScore = currentResult.results[0].similarity_score || 0.0;
        context_info = currentResult.results[0].use_contexts?.[0] || '';
    } else if (currentTab === 'form') {
        id = currentResult.results[0].id;
        originalScore = currentResult.results[0].similarity_score || 0.0;
        context_info = '';
    } else {
        // Handle other tabs (mapping, search, etc.)
        id = currentResult.results?.[0]?.id || currentResult.data?.[0]?.id;
        originalScore = currentResult.results?.[0]?.similarity_score || currentResult.data?.[0]?.similarity_score || 0.0;
        context_info = '';
    }
    // Customize messages based on feedback type
    if (!id) {
        showResult('<div class="error">Unable to identify result for feedback.</div>');
        return;
    }
    const messages = {
        positive: {
            success: 'Thank you for your positive feedback!',
            error: 'Failed to send positive feedback.'
        },
        negative: {
            success: 'Thank you for your negative feedback!',
            error: 'Failed to send negative feedback.'
        }
    };
    
    try {
        const response = await feedbackService.sendFeedback(
            query, id, feedbackType, originalScore, context_info
        );
        
        console.log('Feedback sent successfully:', response);
        showResult(`<div class="success">${messages[feedbackType].success}</div>`);
        
    } catch (error) {
        console.error(`Error sending ${feedbackType} feedback:`, error);
        showResult(`<div class="error">${messages[feedbackType].error}</div>`);
    }
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

