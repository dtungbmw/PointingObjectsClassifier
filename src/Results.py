from datetime import datetime


class PointingPredictionResult:

    def __init__(self):
        self.pointing_unit_vector = None
        self.prob_pointing = None


class ObjectDetectionResult:

    def __init__(self):
        self.detection_results = None


class DepthEstimationResult:

    def __init__(self):
        self.depth_results = None


class JointEstimationResult:

    def __init__(self):
        self.joints_results = None


class ExperimentResult:
    def __init__(self, cfg, device, experiment_name="praxy", timestamp=datetime.now(),  research_question_id ="0", desc="test"):
        self.experiment_name = experiment_name
        self.timestamp = timestamp
        self.device = device
        self.cfg = cfg
        self.cosine_similarity = None
        self.description = desc
        self.normalized_pointing_unit_vector = None
        self.normalized_vector_to_object = None
        self.research_question_id = research_question_id
        self.pointing_unit_vector = None
        self.prob_pointing = None
        self.object_detection_result = None
        self.depth_estimation_result = None
        self.joint_estimation_result = None

    def convert_to_json(self):
        cls_index = 0
        data = {
            "exp": self.experiment_name,
            "researchQuestionId": self.research_question_id,
            "device": self.device,
            "file": self.cfg.movie,
            "sim": self.cosine_similarity.tolist(),
            "description": self.description,
            "prob_pointing": self.prob_pointing,
            "class_name": self.object_detection_result.cls[cls_index].tolist(),
            "norm_pointing_vector": self.normalized_pointing_unit_vector.tolist(),
            "norm_object_vector": self.normalized_vector_to_object.tolist(),
            "timestamp": datetime.now(),
        }

        return data

    def print(self):
        print("(================= ExperimentResult")
        #print(f"==>>> hand_index_2D={hand_index_2D}")
        #print(f"==>>> pointing_unit_vector={pointing_unit_vector}")
        #print(f"==>>> objDetection={objDetection}")
