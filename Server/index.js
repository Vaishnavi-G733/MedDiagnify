const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const AuthRouter = require('./Routes/authRouter');
const productRouter = require('./Routes/productRouter');
require('dotenv').config();
require('./Models/db');

// Ensure required environment variables are set
const requiredEnvVars = ['PORT', 'TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'JWT_SECRET'];
requiredEnvVars.forEach((envVar) => {
    if (!process.env[envVar]) {
        console.error(`Error: Missing environment variable ${envVar}`);
        process.exit(1);
    }
});

const app = express();
const defaultPort = process.env.PORT || 4000;

app.get('/ping', (req, res) => {
    res.send('PONG');
});

app.use(express.json());
app.use(cors());

// Routing
app.use('/auth', (req, res, next) => {
    console.log('Routing middleware hit');
    next();
}, AuthRouter);

app.use('/product', productRouter);

// Centralized error handler
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send({ success: false, message: 'Something went wrong!' });
});

app.listen(defaultPort, () => {
    console.log(`Server is running on port ${defaultPort}`);
});
