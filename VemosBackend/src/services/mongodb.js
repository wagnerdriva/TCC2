const mongoose = require('mongoose');


const uri = `mongodb://${process.env.MONGODB_USER}:${process.env.MONGODB_PASSWORD}@${process.env.MONGODB_HOST}/vehicles?authSource=admin&w=1`;
mongoose.connect(uri);

const vehicleSchema = new mongoose.Schema({
    id: String,
    plate: String,
    model: String,
    color: String
});

module.exports = { mongoose, Vehicle: mongoose.model('Vehicle', vehicleSchema)};