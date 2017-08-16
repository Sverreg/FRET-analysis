// Create gradient image
newImage("Gradient", "8-bit ramp", 256, 256, 1);
run("Select All");
run("Copy");

// Create empty result image
newImage("Result", "RGB black", 256, 256, 1);
run("HSB Stack");

// Paste left-right gradient into B channel
setSlice(3);
run("Paste");

// Rotate and scale gradient
selectWindow("Gradient");
run("Rotate 90 Degrees Right");
run("Multiply...", "value=0.65");
run("Select All");
run("Copy");

// Paste top-bottom scaled gradient into H channel
selectWindow("Result");
setSlice(1);
run("Paste");

// Set saturation (S) channel to all white
setSlice(2);
setForegroundColor(255, 255, 255);
run("Select All");
run("Fill", "slice");

// Convert HSB to RGB
run("RGB Color");