$(document).ready(function() {
    // Load initial visualizations
    updateVisualizations();

    // Upload form submission
    $('#uploadForm').submit(function(e) {
        e.preventDefault();
        let formData = new FormData(this);
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                alert(response.message);
                updateVisualizations();
            },
            error: function(xhr) {
                alert('Error: ' + xhr.responseJSON.error);
            }
        });
    });

    // Function to update visualizations
    function updateVisualizations() {
        $.get('/visualizations', function(data) {
            $('#sentimentPlot').html(`<img src="data:image/png;base64,${data.sentiment_plot}" alt="Sentiment Distribution">`);
            $('#wordCloud').html(`<img src="data:image/png;base64,${data.wordcloud}" alt="Word Cloud">`);
            $('#topWords').html(`<img src="data:image/png;base64,${data.top_words}" alt="Top Words">`);
            $('#reviewLength').html(`<img src="data:image/png;base64,${data.review_length}" alt="Review Length">`);
            $('#sentimentVsLength').html(`<img src="data:image/png;base64,${data.sentiment_vs_length}" alt="Sentiment vs Length">`);
            $('#bigrams').html(`<img src="data:image/png;base64,${data.bigrams}" alt="Bigrams">`);
            $('#trigrams').html(`<img src="data:image/png;base64,${data.trigrams}" alt="Trigrams">`);
            $('#positiveWords').html(`<img src="data:image/png;base64,${data.positive_words}" alt="Positive Words">`);
            $('#negativeWords').html(`<img src="data:image/png;base64,${data.negative_words}" alt="Negative Words">`);
        });
    }
});