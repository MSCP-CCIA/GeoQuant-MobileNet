import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
        """
        s: Factor de escala (magnitud del vector en la hiperesfera). 30.0 es estándar para datasets pequeños/medianos.
        m: Margen angular en radianes. 0.50 fuerza una separación estricta.
        """
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m

        # Pesos (Centroides de las clases)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # 1. Normalización L2 de los embeddings y los pesos (proyección a la esfera unitaria)
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        # 2. Calcular el seno para derivar el ángulo con el margen sumado
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 3. Relajación matemática para evitar inestabilidad si theta + m > pi
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 4. Aplicar el margen SOLO a la clase verdadera (Ground Truth)
        one_hot = torch.zeros(cosine.size(), device=labels.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 5. Escalar y retornar logits para la CrossEntropyLoss
        output *= self.s
        return output