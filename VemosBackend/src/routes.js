const express = require('express');
const Authenticate = require('./controllers/Authenticate');
const Vehicles = require('./controllers/Vehicles');

const routes = new express.Router();

routes.post('/authenticate', Authenticate.authentication);

routes.post('/vehicles', Vehicles.searchVehicles);
routes.post('/vehicles/agg', Vehicles.aggVehicles);
routes.post('/vehicles/count', Vehicles.countVehicles);
routes.get('/vehicles/values/:field', Vehicles.getValues);
routes.get('/vehicles/fields', Vehicles.getFields);

routes.get('/', (req, res) => res.send({ status: 'OK'}));

module.exports = routes;