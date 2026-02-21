// servo driver setup
#include "HCPCA9685.h"
#define  I2CAdd 0x40
HCPCA9685 HCPCA9685(I2CAdd);

// stepper motor setup
#define DIR_PIN  3
#define STEP_PIN 4
#define ENABLE_PIN 5
long targetStep = 0;
const int STEPS_MAX = 800;
#include <AccelStepper.h>
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN); // accurately handles stepper acceleration to prevent missteps


void setup() 
{
  Serial.begin(9600); // setup Serial connection

  // continue servo driver setup
  HCPCA9685.Init(SERVO_MODE);
  HCPCA9685.Sleep(false);

  // continue stepper motor setup
  pinMode(ENABLE_PIN, OUTPUT);
  stepper.setEnablePin(ENABLE_PIN);
  stepper.setPinsInverted(false, false, true); // ENABLE is active LOW
  stepper.disableOutputs();                    // start disabled
  stepper.setMaxSpeed(350);                    // steps/sec
  stepper.setAcceleration(800);                // steps/(sec^2)
  stepper.setMinPulseWidth(15);                // us

  delay(3000);
}


static unsigned long idleStart = 0;
void loop()
{
  stepper.run();

  if (!Serial.available())
    return;

  // check if we're only opening/closing the gripper
  char cmd = Serial.peek();
  if (cmd == 'G') 
  {
    Serial.read(); // consume 'G'
    int gripperAngle = Serial.parseInt();
    if (Serial.read() == '\n') 
    {
      gripperAngle = constrain(gripperAngle, 0, 180);
      int gripperPWM = map(gripperAngle, 0, 180, 10, 450);
      HCPCA9685.Servo(0, gripperPWM);
    }
    return;
  }

  // read 5 servo angles and 1 signed step delta (space-separated)
  int angle1 = Serial.parseInt();      // Gripper
  int angle2 = Serial.parseInt();      // Wrist 2
  int angle3 = Serial.parseInt();      // Wrist 1
  int angle4 = Serial.parseInt();      // Elbow
  int angle5 = Serial.parseInt();      // Shoulders
  int stepSlider = Serial.parseInt();  // Base rotation/Stepper

  if (Serial.read() != '\n')
    return;

  // clamp servo & stepper values for safety
  angle1 = constrain(angle1, 0, 180);
  angle2 = constrain(angle2, 0, 180);
  angle3 = constrain(angle3, 0, 180);
  angle4 = constrain(angle4, 0, 180);
  angle5 = constrain(angle5, 0, 180);
  stepSlider = constrain(stepSlider, 0, 270);

  // convert degree angle to PWM pulse
  /*
    Gripper:   10 -> 450
    Wrist 2:   10 -> 450
    Wrist 1:   10 -> 340
    Elbow:     10 -> 400
    Shoulders: 10 -> 400
  */
  // if servos' movement is inverted invert the angles (e.g. angle1 -> 180 - angle1 and vice versa)
  int gripperPWM   = map(angle1, 0, 180, 10, 450);
  int wrist2PWM    = map(angle2, 0, 180, 10, 450);
  int wrist1PWM    = map(180 - angle3, 0, 180, 10, 340);
  int elbowPWM     = map(angle4, 0, 180, 10, 400);
  int shoulderPWM  = map(180 - angle5, 0, 180, 10, 400);
  // mirrored shoulder (servo 6)
  int shoulderMirrorPWM = map(angle5, 0, 180, 10, 400);
  // map stepper slider to absolute step position
  targetStep = map(stepSlider, 0, 270, 0, STEPS_MAX);

  // send servo PWM pulses to PCA9685 (the servo driver module)
  HCPCA9685.Servo(0, gripperPWM);
  HCPCA9685.Servo(1, wrist2PWM);
  HCPCA9685.Servo(2, wrist1PWM);
  HCPCA9685.Servo(3, elbowPWM);
  HCPCA9685.Servo(4, shoulderPWM);
  HCPCA9685.Servo(5, shoulderMirrorPWM);

  // moves the stepper motor to its target position
  if (targetStep != stepper.targetPosition())
  {
    stepper.enableOutputs();
    stepper.moveTo(targetStep);
  }
  
  // stops the stepper motor after it has moved to the target position
  if (stepper.distanceToGo() == 0)
  {
    if (idleStart == 0)
      idleStart = millis();
    if (millis() - idleStart > 100) // 100 ms idle
      stepper.disableOutputs();
  }
  else
    idleStart = 0;
}
// use this to test whether the stepper motor is behaving correctly, if it moves all 270 degrees then it's correct
// to reset the stepper motor to 0 either manually rotate the motor or set stepper.moveTo(-800);
/*void loop()
{
  stepper.moveTo(800);

  if (stepper.distanceToGo() != 0)
  {
    stepper.enableOutputs();
    stepper.run();
  }
  else
    stepper.disableOutputs();
}*/