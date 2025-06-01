const chatDiv = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');

let history = [];

sendBtn.onclick = async () => {
  const user = inputEl.value.trim();
  if (!user) return;

  appendMessage('You', user);
  inputEl.value = '';

  const resp = await fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ user, history })
  });
  const { reply, history: newHistory } = await resp.json();
  history = newHistory;

  appendMessage('MirAI', reply);
};

function appendMessage(author, text) {
  const p = document.createElement('p');
  p.innerHTML = `<strong>${author}:</strong> ${text}`;
  chatDiv.appendChild(p);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}
