function displayfhirv2Result(result, dataset) {      
    let formObjects = result;      
 
    
    return `<div class="success">     
        <h2>Best domain match from ${dataset} dataset: ${formObjects.results[0]}</h2>     
 
    </div>`;  
}  

window.displayfhirv2Result = displayfhirv2Result;