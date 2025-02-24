// const nodemailer = require('nodemailer');

// const transporter = nodemailer.createTransport({
//     service: 'Gmail',
//     auth: {
//         user: process.env.AUTH_EMAIL,
//         pass: process.env.AUTH_PASSWORD,
//     },
// });

// module.exports = transporter;
const nodemailer = require('nodemailer');

const transporter = nodemailer.createTransport({
    service: "gmail",
    host: "smtp.gmail.com",
    port: 465,
    secure: true,
    auth: {
        user: "saisrimayinuli@gmail.com",
        pass: "cegl opzl wwat cyzy",
    },
});

module.exports = transporter;

