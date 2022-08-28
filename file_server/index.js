const fs = require('fs')
const multer = require('multer');
const express = require('express');

const app = express();

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const { id } = req.params
    const path = `./public/${id}`
    fs.mkdirSync(path, { recursive: true })
    cb(null, path)
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname) //Appending extension
  }
})

const upload = multer({ storage: storage });

app.use('/vemos/data', express.static('public'));

// Configuramos o upload como um middleware que
// espera um arquivo cujo a chave é "foto"
app.post('/vemos/data/upload/:id', upload.single('image'), (req, res) => {
    res.json({ status: "OK" });
});


app.listen(3002, function () {
    console.log('Listening on http://localhost:3000/');
});
