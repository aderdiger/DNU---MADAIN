function makePrediction() {
    // Get the uploaded file from the file input element
    const fileInput = document.getElementById('uploaded_file');
    const uploadedFile = fileInput.files[0];

    // Create a FormData object to send the file to the server
    const formData = new FormData();
    formData.append('uploaded_file', uploadedFile);

    // Make a POST request to the Streamlit backend
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Get the prediction from the response
        const prediction = data.prediction;

        // Display the prediction in the Streamlit app
        Streamlit.setComponentValue(`Prediction: ${prediction}`);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}





