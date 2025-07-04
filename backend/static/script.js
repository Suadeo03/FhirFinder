// Tab switching functionality
let currentTab = 'resources';

function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to clicked tab button
    event.target.classList.add('active');
    
    // Update current tab
    currentTab = tabName;
    
    // Clear previous results when switching tabs
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
                case 'patients':
                    endpoint = '/api/v1/form/';
                    break;
                case 'clinical':
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
            showResult(`<div class="error">❌ No results found in ${currentTab} dataset</div>`);
        }
    } catch (error) {
        console.error('Frontend error:', error);
        showResult(`<div class="error">❌ Error searching ${currentTab}: ${error.message}</div>`);
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
    return `<div class="success">
    <h2>Best match from ${dataset} dataset: ${JSON.stringify(result.results[0].resource_type)}</h2><br/>
    ${JSON.stringify(result.results[0].fhir_resource[0].text.div)}<br/>
    <details>
    <summary>Description</summary>
    ${JSON.stringify(result.results[0].description)}
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

function clearResult() {
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';
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

function testConnection() {
    showResult('<div class="loading">🔄 Testing connection...</div>');
}

// Allow Enter key to perform lookup in any active input
document.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        const activeInput = document.querySelector('.tab-content.active input');
        if (activeInput === document.activeElement) {
            lookupBtn();
        }
    }
});

// Test connection on page load
window.addEventListener('DOMContentLoaded', async () => {
    setTimeout(testConnection, 500);
});