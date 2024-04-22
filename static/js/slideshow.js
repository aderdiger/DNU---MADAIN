// Initialize the slide index to show the first slide initially
var slideIndex = 1;
showSlides(slideIndex);

// Function to move to the next or previous slide
function plusSlides(n) {
  showSlides(slideIndex += n);
}

// Function to move to a slide specified by the index number
function currentSlide(n) {
  showSlides(slideIndex = n);
}

// Main function to display the appropriate slide and highlight the corresponding dot
function showSlides(n) {
    var i;
    var slides = document.getElementsByClassName("slide");  // Get all elements with class "slide"
    var dots = document.getElementsByClassName("dot");      // Get all elements with class "dot"
    
    // If the slide index is greater than the number of slides, reset to the first slide
    if (n > slides.length) {slideIndex = 1}
    // If the slide index is less than 1, set it to the last slide
    if (n < 1) {slideIndex = slides.length}

    // Hide all slides
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }

    // Remove the "active" class from all dots
    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }

    // Display the current slide and add "active" class to the corresponding dot
    slides[slideIndex-1].style.display = "block";
    dots[slideIndex-1].className += " active";
}

// Add event listener for keyboard controls
document.addEventListener('keydown', function(event) {
    // Navigate to the previous slide if the left arrow key is pressed
    if (event.key === "ArrowLeft") {
        plusSlides(-1);
    } 
    // Navigate to the next slide if the right arrow key is pressed
    else if (event.key === "ArrowRight") {
        plusSlides(1);
    }
});

