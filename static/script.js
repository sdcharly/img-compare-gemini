// Event listener for the search form submission
document.getElementById('searchForm').addEventListener('submit', function(e) {
    e.preventDefault();
    processImageRequest(this, '/search', 'searchResults', true);
});

// Event listener for the upload form submission
document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    processImageRequest(this, '/upsert', 'uploadStatus', false);
});

// Function to process the image request
function processImageRequest(form, url, resultContainerId, isSearch) {
    var formData = new FormData(form);
    var resultContainer = document.getElementById(resultContainerId);
    var progressBar = document.querySelector('.progress-bar');
    var progressDiv = document.querySelector('.progress');

    resultContainer.innerHTML = ''; // Clear previous results or status
    progressBar.style.width = '0%';
    progressDiv.style.display = 'block';

    fetch(url, {
        method: 'POST',
        body: formData
    }).then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error || 'Network response was not ok') });
        }
        return isSearch ? response.json() : response.text();
    }).then(data => {
        progressBar.style.width = '100%';
        if (isSearch) {
            displaySearchResults(data, resultContainer);
        } else {
            resultContainer.innerHTML = data;
        }
    }).catch(error => {
        console.error('Fetch error:', error);
        resultContainer.innerText = error.message || 'Operation failed. Please try again.';
    }).finally(() => {
        setTimeout(() => progressDiv.style.display = 'none', 2000); // Hide progress bar after delay
    });
}

// Function to display search results
function displaySearchResults(data, container) {
    if (data.results && data.results.length > 0) {
        var resultsHtml = data.results.map(result => {
            return `<div><strong>ID:</strong> ${result.id}, <strong>Score:</strong> ${result.score}</div>`;
        }).join('');
        container.innerHTML = `<h4>Results:</h4>${resultsHtml}`;
    } else {
        container.innerHTML = 'No matches found';
    }
}
