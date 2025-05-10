function showSection(id) {
    document.getElementById("generate").style.display = id === 'generate' ? 'block' : 'none';
    document.getElementById("modify").style.display = id === 'modify' ? 'block' : 'none';
  }

  document.getElementById("generateForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const prompt = document.getElementById("prompt").value;
    const style = document.getElementById("style").value;

    const response = await fetch("https://1c60-35-197-48-70.ngrok-free.app", {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, style })
    });
    const data = await response.json();

    const preview = document.getElementById("generatePreview");
    preview.innerHTML = "";
    data.images.forEach(src => {
      const img = document.createElement("img");
      img.src = src;
      preview.appendChild(img);
    });

    const slider = document.getElementById("generateSlider");
    slider.innerHTML = "";
    data.images.forEach(src => {
      const img = document.createElement("img");
      img.src = src;
      slider.appendChild(img);
    });
  });

  document.getElementById("modifyForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const formData = new FormData();
    formData.append("image", document.getElementById("upload").files[0]);
    formData.append("prompt", document.getElementById("modPrompt").value);
    formData.append("model", document.getElementById("model").value);

    const response = await fetch("https://1c60-35-197-48-70.ngrok-free.app", {
      method: "POST",
      body: formData
    });
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const img = document.createElement("img");
    img.src = url;
    document.getElementById("modifyPreview").innerHTML = "";
    document.getElementById("modifyPreview").appendChild(img);

    const slider = document.getElementById("modifySlider");
    slider.innerHTML = "";
    const modifiedImage = document.createElement("img");
    modifiedImage.src = url;
    slider.appendChild(modifiedImage);
  });