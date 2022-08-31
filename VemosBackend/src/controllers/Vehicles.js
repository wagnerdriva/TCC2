const { mongoose, Vehicle } = require("../services/mongodb")

module.exports = {
    async searchVehicles(req, res){
        const { filter } = req.body;

        const result = await Vehicle.find( filter ).select("_id id plate color model category brand createdAt possibleImages");

        res.send(result);
    },
    async aggVehicles(req, res){
        const { filter, field } = req.body;

        if(filter && filter.createdAt){
            filter.createdAt["$gte"] = new Date(filter.createdAt["$gte"])
            filter.createdAt["$lt"] = new Date(filter.createdAt["$lt"])
        }

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

        const result = await Vehicle.countDocuments(filter);
        res.send({count: result});
    },
    async getFields(req, res){
        const data = [
            { label: "Placa", field: "plate"},
            { label: "Categoria", field: "category"},
            { label: "Marca", field: "brand"},
            { label: "Modelo", field: "model"},
            { label: "Cor", field: "color"},
            { label: "Data", field: "createdAt"}
        ]

        res.send(data);
    },
    async getValues(req, res){
        const { field, filter } = req.body;

        let response = []
        if(field)
            response = await Vehicle.distinct(field, filter);
            
        res.send(response);
    },
}