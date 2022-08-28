const { mongoose, Vehicle } = require("../services/mongodb")

module.exports = {
    async searchVehicles(req, res){
        const { filter } = req.body;

        const result = await Vehicle.find( filter ).select("_id id plate color model category brand createdAt possibleImages");

        res.send(result);
    },
    async aggVehicles(req, res){
        const { filter, field } = req.body;


        const result = await Vehicle
            .aggregate([
                {"$match" : filter ? filter : {}},
                {"$group" : {_id: `$${field}`, count: { $sum:1 } }}
            ])

        const response = result.map(elem => {
            return {x: elem._id, y: elem.count}
        })

        res.send(response);
    },
    async countVehicles(req, res){
        const { filter } = req.body;

        const result = await Vehicle.count(filter);
        res.send({count: result});
    },
    async getFields(req, res){
        const data = [
            { label: "Placa", field: "plate"},
            { label: "Categoria", field: "category"},
            { label: "Marca", field: "brand"},
            { label: "Modelo", field: "model"},
            { label: "Cor", field: "color"}
        ]

        res.send(data);
    },
    async getValues(req, res){
        const { field } = req.params;

        const response = await Vehicle.distinct(field);
        res.send(response);
    },
}