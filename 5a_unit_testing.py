import unittest
from EmotionDetection import emotion_detector

class TestEmotionDetector(unittest.TestCase):

    def test_anger(self):
        result = emotion_detector('I am really mad about this')
        self.assertEqual(result['dominant_emotion'], 'anger')

    def test_joy(self):
        result = emotion_detector('I am feeling great today!')
        self.assertEqual(result['dominant_emotion'], 'joy')

    def test_sadness(self):
        result = emotion_detector('I am very sad today')
        self.assertEqual(result['dominant_emotion'], 'sadness')

    def test_disgust(self):
        result = emotion_detector('I feel disgusted and revolted by that behavior')
        self.assertEqual(result['dominant_emotion'], 'disgust')

    def test_fear(self):
        result = emotion_detector('I am terrified of what might happen next')
        self.assertEqual(result['dominant_emotion'], 'fear')

if __name__ == '__main__':
    unittest.main()
