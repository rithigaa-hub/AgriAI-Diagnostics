console.log("AgriAI Diagnostics Loaded");



const fileInput=document.querySelector(
    'input[type="file"]'
);



if(fileInput)
{

fileInput.addEventListener(
"change",
function(){

console.log(
"Image Selected:",
this.files[0].name
);


});


}
