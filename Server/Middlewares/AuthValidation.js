const Joi = require('joi');

const signupValidation = (req, res, next) => {
    const schema = Joi.object({
        name: Joi.string().min(3).max(100).required(),
        email: Joi.string().email().required(),
        password: Joi.string().min(4).max(100).required(),
        dob: Joi.string().required(),
        phoneotp: Joi.string().required(),
        emailotp:Joi.string().required(),
        phone: Joi.string().length(10).pattern(/^[0-9]+$/).required() // Phone number validation
    });

    const { error } = schema.validate(req.body);
    if (error) {
        console.log(error);
        return res.status(400).json({ message: "Bad request", error: error.details[0].message });
    }
    next();
};

const loginValidation = (req, res, next) => {
    const schema = Joi.object({
        email: Joi.string().email().required(),
        password: Joi.string().min(4).max(100).required()
    });

    const { error } = schema.validate(req.body);
    if (error) {
        return res.status(400).json({ message: "Bad request", error });
    }
    next();
};

module.exports = {
    signupValidation,
    loginValidation
};
