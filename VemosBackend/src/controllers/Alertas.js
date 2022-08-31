const { Alerta } = require("../services/mongodb")

module.exports = {
    async listAlertas(req, res){
        const result = await Alerta.find();

        res.send(result);
    },
    async createAlerta(req, res){
        const { alerta } = req.body;

        if(!alerta.email)
            res.status(404).send("Oh uh, something went wrong");
        else {
            const newAlerta = new Alerta(alerta);
            newAlerta.save();
            res.send(newAlerta);
        }
    },
    async deleteAlerta(req, res){
        const { _id } = req.body;

        try {
            await Alerta.deleteOne({ _id })
        } catch (error) {
            console.log(error);
        }

        res.send({status: "OK"});
    }
}