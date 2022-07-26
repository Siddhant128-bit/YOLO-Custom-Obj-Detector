// Paste this code on the console of the collab page it will simulate somone clicking on the page every 60000 ms 
function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)