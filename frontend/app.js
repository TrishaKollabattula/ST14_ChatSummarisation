function summarizeChat() {
    let chatText = document.getElementById("chatInput").value;

    fetch("http://127.0.0.1:5000/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_text: chatText })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("summaryOutput").innerText = data.summary;
    })
    .catch(error => console.error("Error:", error));
}
