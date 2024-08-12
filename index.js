import express from "express";
import cors from "cors";
import { execSync } from "child_process";

const app = express();

app.use(cors());
app.use(express.json({ type: "application/json" }));

app.get("/healthcheck", (req, res) => {
    res.status(200).json("healthcheck");
});

app.post("/predict", (req, res) => {
    try {
        console.log(req.body);
        const { knowledge_level, learning_rate, error_rate } = req.body;
        console.log(
            `python3 model.py --state ${knowledge_level} ${learning_rate} ${error_rate}`
        );
        const output = execSync(
            `python3 model.py --state ${knowledge_level} ${learning_rate} ${error_rate}`
        );

        const question_knowledge_level = output.toString().trim();
        res.status(200).send({ question_knowledge_level });
    } catch (err) {
        console.log(err);
    }
});

app.listen(8080, async () => {
    console.log("Listening on port 8080");
});
