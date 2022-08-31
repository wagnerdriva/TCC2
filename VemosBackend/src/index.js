require('dotenv').config();

const express = require('express');
const cors = require('cors');

const app = express(); // Aplicacao criada com express, basicamente um servidor.
app.use(express.json());

app.use(cors()); // Habilita o acesso de qualquer aplicacao a esse backend

app.use('/vemos/backend', require('./routes')); // Ativa todas as rotas que estao definidas no arquivo routes.js

app.listen(8080); // Porta onde ficara o servidor, 3333 para desenvolvimento.