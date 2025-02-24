const mongoose = require('mongoose');

const otpVerificationSchema = new mongoose.Schema({
    email: {
        type: String,
        required: true,
        unique: true,
    },
    otp: {
        type: String,
        required: true,
    },
    expiresat: {
        type: Date,
        required: true,
    },
    createdat: {
        type: Date,
        default: Date.now,
    },
});

const Otpverification = mongoose.model('Otpverification', otpVerificationSchema);

module.exports = Otpverification;
