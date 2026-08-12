#include <Servo.h>

Servo servo1;
Servo servo2;
int rf;
int lf;

void setup() {
  Serial.begin(115200);
  servo1.attach(9);
  servo2.attach(10);
  
  // Stop both servos initially
  stop();
  rf = 71;
  lf = 180;
  
  Serial.println("Arduino ready for commands!");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.length() > 0) {
      parseCommand(command);
    }
  }
}

void parseCommand(String cmd) {
  int spaceIdx = cmd.indexOf(' ');
  if (spaceIdx == -1) return;
  
  String direction = cmd.substring(0, spaceIdx);
  direction.toUpperCase();
  
  String magStr = cmd.substring(spaceIdx + 1);
  int magnitude = magStr.toInt();
  
  Serial.print("Executing: ");
  Serial.print(direction);
  Serial.print(" ");
  Serial.println(magnitude);
  
  if (direction == "FORWARD") {
    forward(magnitude);
  } else if (direction == "BACKWARD" || direction == "BACK") {
    back(magnitude);
  } else if (direction == "LEFT") {
    left(magnitude);
  } else if (direction == "RIGHT") {
    right(magnitude);
  } else {
    Serial.println("Unknown direction.");
  }
}

// === EXISTING MOTOR TUNING LOGIC ===

void forward(int a) {
  servo1.write(rf); // Full speed clockwise
  servo2.write(lf);
  delay(a * 2800);
  stop();
}

void stop() {
  servo1.write(89);
  servo2.write(94);
}

void left(int a) {
  servo2.write(94);
  servo1.write(rf);
  delay(a * 31);
  stop();
}

void right(int a) {
  servo1.write(89);
  servo2.write(lf);
  delay(a * 29.1);
  stop();
}

void back(int a) {
  servo1.write(105);
  servo2.write(0);
  delay(a * 2700);
  stop();
}
