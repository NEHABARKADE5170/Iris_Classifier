const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('file');
const fileName = document.getElementById('fileName');
const uploadText = document.getElementById('uploadText');

// Click to upload
dropZone.addEventListener('click', () => {
    fileInput.click();
});

// File selected via click
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) {
        fileName.textContent = '✅ ' + file.name;
        uploadText.textContent = file.name;
    }
});

// Drag over
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('dragover');
});

// Drag leave
dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');
});

// Drop
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');

    const file = e.dataTransfer.files[0];

    if (file && file.name.endsWith('.csv')) {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;

        fileName.textContent = '✅ ' + file.name;
        uploadText.textContent = file.name;
    } else {
        fileName.textContent = '❌ Please drop a CSV file!';
    }
});