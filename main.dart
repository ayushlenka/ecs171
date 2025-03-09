import 'package:flutter/material.dart';

void main() {
  runApp(const ECS171App());
}

class ECS171App extends StatelessWidget {
  const ECS171App({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'ECS 171 Group 10 - ML Project',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TextEditingController inputController = TextEditingController();
  String result = "";
  String selectedCompany = "Amazon";
  final List<String> companies = ["Amazon", "Microsoft", "Tesla", "Apple"];

  void handlePrediction() {
    // TODO: Replace with actual ML API call
    setState(() {
      result = "Prediction result: Sample Output for $selectedCompany";
    });
  }

  void handleTweet() {
    // TODO: Send the tweet to LLM API
    setState(() {
      result = "Tweet sent for $selectedCompany: ${inputController.text}";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ECS 171 Group 10 - ML Predictor'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const Text(
              'Select a Company:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            DropdownButton<String>(
              value: selectedCompany,
              onChanged: (String? newValue) {
                setState(() {
                  selectedCompany = newValue!;
                });
              },
              items:
                  companies.map<DropdownMenuItem<String>>((String value) {
                    return DropdownMenuItem<String>(
                      value: value,
                      child: Text(value),
                    );
                  }).toList(),
            ),
            const SizedBox(height: 20),
            const Text(
              'Enter data for prediction:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: inputController,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                hintText: 'Enter input data',
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: handlePrediction,
              child: const Text('Next'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(onPressed: handleTweet, child: const Text('Tweet')),
            const SizedBox(height: 20),
            Text(
              result,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }
}
