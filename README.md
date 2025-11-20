# ARQUITECTURA DEL SISTEMA
## Sistema Inteligente basado en Deep Learning y Análisis Multimodal para la Detección Temprana de Heliothis virescens en Cultivos de Arándanos en Virú

---

## 1. PROPUESTA DE SOLUCIÓN

### Descripción del Sistema

El sistema propuesto es una **plataforma inteligente de detección temprana de plagas** que combina visión por computadora, aprendizaje profundo y análisis multimodal para identificar *Heliothis virescens* en estadios iniciales en cultivos de arándanos de Virú, La Libertad.

**Problema que Soluciona:**
- Detección tardía de plagas que ocasiona pérdidas del 30-40% en la cosecha
- Dependencia de inspecciones manuales con alta tasa de error
- Uso excesivo de agroquímicos por aplicación preventiva generalizada
- Falta de datos en tiempo real para toma de decisiones

**Aporte Innovador:**
- **Detección Temprana**: Identificación de plagas en estadios larvarios iniciales antes de daños visibles
- **Análisis Multimodal**: Fusión de imágenes RGB con datos ambientales (temperatura, humedad, fenología del cultivo)
- **Modelos Optimizados**: Arquitecturas CNN ligeras con mecanismos de atención para despliegue en edge devices
- **Sistema Integral**: Desde captura en campo hasta alertas en tiempo real y dashboard de gestión

**Dirigido a:**
- Productores de arándanos de exportación en Virú
- Empresas agroexportadoras como Fagro Latinoamérica SAC
- Ingenieros agrónomos y responsables de sanidad vegetal

**Tecnologías Principales:**

**Deep Learning:**
- PyTorch 2.5+ (Framework principal)
- YOLOv8/YOLOv9 optimizado con módulos CBAM (Convolutional Block Attention Module)
- TensorFlow Lite para inferencia en edge
- OpenCV para preprocesamiento de imágenes
- Ultralytics para detección de objetos

**Backend & API:**
- FastAPI (Python) para endpoints de inferencia
- Spring Boot (Java) opcional para lógica de negocio empresarial
- Node.js con Express para servicios en tiempo real
- WebSocket para comunicación bidireccional

**Frontend:**
- React 18+ con TypeScript
- Material-UI/Tailwind CSS para UI responsive
- Progressive Web App (PWA) para acceso móvil
- Dashboard interactivo con Chart.js/Recharts

**Base de Datos:**
- PostgreSQL 16 (datos estructurados, históricos)
- MongoDB (imágenes metadata, logs)
- Redis para caché y colas de procesamiento
- Amazon S3/Azure Blob para almacenamiento de imágenes

**Cloud & Deployment:**
- AWS/Azure/GCP (híbrido según benchmarking)
- Docker + Kubernetes para orquestación
- NVIDIA Jetson Nano/Xavier NX para edge computing
- CI/CD con GitHub Actions/GitLab CI

**IoT & Sensores:**
- Sensores ambientales (DHT22, BME280)
- Cámaras Raspberry Pi HQ/ESP32-CAM
- Protocolo MQTT para comunicación IoT
- Edge computing con TensorFlow Lite/ONNX Runtime

---

## 2. DIAGRAMA ARQUITECTÓNICO GENERAL

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CAPA DE USUARIOS                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Agricultor  │  │  Agrónomo    │  │ Supervisor   │  │Administrador │   │
│  │   (Móvil)    │  │  (Tablet)    │  │   (Laptop)   │  │   (Desktop)  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│           │                │                 │                  │            │
└───────────┼────────────────┼─────────────────┼──────────────────┼────────────┘
            │                │                 │                  │
            └────────────────┴─────────────────┴──────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CAPA DE PRESENTACIÓN (PWA)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Frontend - React 18 + TypeScript                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  Dashboard   │  │ Detecciones  │  │   Reportes   │              │   │
│  │  │  Analytics   │  │  en Tiempo   │  │  Históricos  │              │   │
│  │  │              │  │     Real     │  │              │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │                                                                      │   │
│  │  Tecnologías: Material-UI, Recharts, Leaflet Maps, React Query     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CAPA DE API GATEWAY                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    NGINX / Kong API Gateway                          │   │
│  │  • Autenticación JWT                                                 │   │
│  │  • Rate Limiting                                                     │   │
│  │  • Load Balancing                                                    │   │
│  │  • SSL/TLS Termination                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CAPA DE LÓGICA DE NEGOCIO                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  FastAPI         │  │  Node.js +       │  │  Spring Boot     │         │
│  │  (Inferencia)    │  │  Express         │  │  (Empresarial)   │         │
│  │                  │  │  (WebSocket)     │  │  (Opcional)      │         │
│  │ • Upload Images  │  │ • Real-time      │  │ • User Mgmt     │         │
│  │ • Model Predict  │  │   Notifications  │  │ • Reporting     │         │
│  │ • Batch Process  │  │ • Live Alerts    │  │ • Analytics     │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                              │
│  Microservicios:                                                             │
│  • Auth Service (Autenticación/Autorización)                                │
│  • Detection Service (Inferencia de modelos)                                │
│  • Environmental Service (Datos multimodales)                               │
│  • Notification Service (Alertas y notificaciones)                          │
│  • Analytics Service (Reportes y estadísticas)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CAPA DE INTELIGENCIA ARTIFICIAL                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Deep Learning Pipeline                            │   │
│  │  ┌────────────────────────────────────────────────────────────┐     │   │
│  │  │  1. Preprocesamiento                                        │     │   │
│  │  │     • Resize, Normalize, Augmentation                      │     │   │
│  │  │     • OpenCV, Albumentations                               │     │   │
│  │  └────────────────────────────────────────────────────────────┘     │   │
│  │  ┌────────────────────────────────────────────────────────────┐     │   │
│  │  │  2. Modelo de Detección (PyTorch)                          │     │   │
│  │  │     Arquitectura: YOLOv8 + CBAM                            │     │   │
│  │  │     • Backbone: CSPDarknet53 optimizado                    │     │   │
│  │  │     • Neck: PANet con BiFPN                                │     │   │
│  │  │     • Head: Detección con atención espacial                │     │   │
│  │  │     • Precisión: >95% mAP                                  │     │   │
│  │  │     • Inferencia: ~45ms por imagen                         │     │   │
│  │  └────────────────────────────────────────────────────────────┘     │   │
│  │  ┌────────────────────────────────────────────────────────────┐     │   │
│  │  │  3. Fusión Multimodal                                      │     │   │
│  │  │     • Features CNN + Datos Ambientales                     │     │   │
│  │  │     • MLP para clasificación final                         │     │   │
│  │  │     • Temperatura, Humedad, Fenología                      │     │   │
│  │  └────────────────────────────────────────────────────────────┘     │   │
│  │  ┌────────────────────────────────────────────────────────────┐     │   │
│  │  │  4. Post-procesamiento                                     │     │   │
│  │  │     • NMS (Non-Maximum Suppression)                        │     │   │
│  │  │     • Filtrado por confianza (>0.7)                        │     │   │
│  │  │     • Tracking multi-frame (DeepSORT)                      │     │   │
│  │  └────────────────────────────────────────────────────────────┘     │   │
│  │                                                                      │   │
│  │  Frameworks: PyTorch, Ultralytics, TorchVision, ONNX Runtime        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CAPA DE DATOS                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  PostgreSQL 16   │  │   MongoDB 7      │  │   Redis 7        │         │
│  │                  │  │                  │  │                  │         │
│  │ • Usuarios       │  │ • Imágenes       │  │ • Cache          │         │
│  │ • Detecciones    │  │   Metadata       │  │ • Sesiones       │         │
│  │ • Alertas        │  │ • Logs           │  │ • Cola Tareas    │         │
│  │ • Históricos     │  │ • Timestamps     │  │ • Pub/Sub        │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │               Almacenamiento de Archivos                             │   │
│  │  • AWS S3 / Azure Blob Storage / Google Cloud Storage               │   │
│  │  • Imágenes originales y procesadas                                 │   │
│  │  • Modelos entrenados (.pt, .onnx)                                  │   │
│  │  • Reportes PDF/Excel exportados                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CAPA DE ADQUISICIÓN (EDGE)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Edge Computing Devices                            │   │
│  │  ┌────────────────────────────────────────────────────────────┐     │   │
│  │  │  NVIDIA Jetson Nano/Xavier NX                              │     │   │
│  │  │  • TensorFlow Lite / ONNX Runtime                          │     │   │
│  │  │  • Inferencia local (~20 FPS)                              │     │   │
│  │  │  • Modo offline con sincronización                         │     │   │
│  │  └────────────────────────────────────────────────────────────┘     │   │
│  │                                                                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │   │
│  │  │ Raspberry Pi 4 │  │   ESP32-CAM    │  │  Sensores IoT  │        │   │
│  │  │  + HQ Camera   │  │   (Low Cost)   │  │                │        │   │
│  │  │                │  │                │  │ • DHT22 (T/H)  │        │   │
│  │  │ • Captura 4K   │  │ • WiFi/BLE     │  │ • BME280       │        │   │
│  │  │ • USB/PoE      │  │ • 2MP images   │  │ • Soil Sensor  │        │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘        │   │
│  │                                                                      │   │
│  │  Protocolo de Comunicación: MQTT (Mosquitto Broker)                 │   │
│  │  Energía: Solar + Batería (opcional para zonas remotas)             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INFRAESTRUCTURA CLOUD (AWS/Azure/GCP)                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Servicios de Cloud Computing                                       │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │   │
│  │  │  AWS EC2/      │  │  AWS Lambda/   │  │   AWS S3/      │        │   │
│  │  │  Azure VM/     │  │  Azure Func/   │  │   Azure Blob/  │        │   │
│  │  │  GCP Compute   │  │  Cloud Run     │  │   GCS          │        │   │
│  │  │                │  │                │  │                │        │   │
│  │  │ • GPU T4/V100  │  │ • Serverless   │  │ • Object Store │        │   │
│  │  │ • Auto-scaling │  │ • Event-driven │  │ • CDN delivery │        │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘        │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────┐     │   │
│  │  │  Container Orchestration: Kubernetes (EKS/AKS/GKE)        │     │   │
│  │  │  • Docker containers                                       │     │   │
│  │  │  • Helm charts                                             │     │   │
│  │  │  • Horizontal Pod Autoscaler                               │     │   │
│  │  └────────────────────────────────────────────────────────────┘     │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────┐     │   │
│  │  │  Monitoring & Logging                                      │     │   │
│  │  │  • Prometheus + Grafana (métricas)                         │     │   │
│  │  │  • ELK Stack (logs centralizados)                          │     │   │
│  │  │  • AWS CloudWatch / Azure Monitor / GCP Logging            │     │   │
│  │  └────────────────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SEGURIDAD Y CUMPLIMIENTO                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • SSL/TLS (HTTPS)                                                   │   │
│  │  • OAuth 2.0 + JWT Authentication                                    │   │
│  │  • RBAC (Role-Based Access Control)                                  │   │
│  │  • Encriptación en tránsito y reposo (AES-256)                       │   │
│  │  • Compliance: ISO 27001, SOC 2, GDPR                                │   │
│  │  • Auditoría y trazabilidad de acciones                              │   │
│  │  • Firewall + WAF (Web Application Firewall)                         │   │
│  │  • VPN para acceso remoto seguro                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. DIAGRAMA DE FLUJO DEL SISTEMA

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      FLUJO DE DETECCIÓN DE PLAGAS                             │
└──────────────────────────────────────────────────────────────────────────────┘

    [INICIO]
       │
       ▼
┌─────────────────┐
│  1. CAPTURA     │  ◄──── Raspberry Pi 4 + HQ Camera
│     DE IMAGEN   │         ESP32-CAM
└─────────────────┘         Smartphone (PWA)
       │
       ▼
┌─────────────────┐
│  2. METADATA    │  ◄──── GPS coordinates
│     CONTEXTUAL  │         Timestamp
└─────────────────┘         Lote/parcela
       │                    Fenología
       ▼
┌─────────────────┐
│  3. ADQUISICIÓN │  ◄──── Sensores DHT22/BME280
│     DATOS       │         Temperatura
│     AMBIENTALES │         Humedad relativa
└─────────────────┘         Presión atmosférica
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  4. ENVÍO AL SERVIDOR                                    │
│     • Protocolo: MQTT / HTTP REST API                   │
│     • Compresión JPEG (85% quality)                     │
│     • Batch processing si no hay conectividad           │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  5. COLA DE PROCESAMIENTO (Redis)                       │
│     • Priority queue basado en urgencia                 │
│     • Deduplicación de imágenes                         │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  6. PREPROCESAMIENTO                                     │
│     • Resize → 640x640 pixels                           │
│     • Normalización RGB → [0,1]                         │
│     • Eliminación de ruido (Gaussian blur)              │
│     • Ajuste de brillo/contraste (CLAHE)                │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  7. INFERENCIA - MODELO DE DEEP LEARNING (YOLOv8 + CBAM)        │
│                                                                   │
│     INPUT: Imagen 640x640x3                                      │
│        │                                                          │
│        ▼                                                          │
│     ┌──────────────────────────────────┐                        │
│     │  Backbone (CSPDarknet53)         │                        │
│     │  • Conv layers with CBAM         │                        │
│     │  • Feature extraction            │                        │
│     │  • Output: [P3, P4, P5]          │                        │
│     └──────────────────────────────────┘                        │
│        │                                                          │
│        ▼                                                          │
│     ┌──────────────────────────────────┐                        │
│     │  Neck (PANet + BiFPN)            │                        │
│     │  • Multi-scale fusion            │                        │
│     │  • Bidirectional connections     │                        │
│     └──────────────────────────────────┘                        │
│        │                                                          │
│        ▼                                                          │
│     ┌──────────────────────────────────┐                        │
│     │  Detection Head                  │                        │
│     │  • Bounding boxes                │                        │
│     │  • Confidence scores             │                        │
│     │  • Class probabilities           │                        │
│     └──────────────────────────────────┘                        │
│        │                                                          │
│     OUTPUT: [x, y, w, h, conf, class]                           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  8. FUSIÓN MULTIMODAL                                    │
│     • Features CNN (1024-dim) + Datos ambientales (5-d) │
│     • MLP (1029 → 512 → 256 → 2)                        │
│     • Clasificación: Positivo/Negativo                  │
│     • Confianza ajustada                                │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  9. POST-PROCESAMIENTO                                   │
│     • NMS (IoU threshold = 0.45)                        │
│     • Filtrado por confianza (> 0.7)                    │
│     • Clasificación estadio larval                      │
│     • Conteo de individuos                              │
└─────────────────────────────────────────────────────────┘
       │
       ▼
    ┌─────────────┐
    │ ¿Detección? │
    └─────────────┘
       │         │
     SÍ│         │NO
       │         │
       ▼         ▼
┌─────────────┐  ┌─────────────────────────┐
│ 10. ALERTA  │  │ 11. ALMACENAMIENTO     │
│    GENERADA │  │     HISTÓRICO          │
└─────────────┘  └─────────────────────────┘
       │                    │
       ▼                    │
┌─────────────────────────────────────────────┐
│ 12. NOTIFICACIÓN MULTI-CANAL                │
│     • Push notification (móvil)             │
│     • Email                                 │
│     • SMS (crítico)                         │
│     • Dashboard en tiempo real              │
└─────────────────────────────────────────────┘
       │                    │
       └─────────┬──────────┘
                 ▼
┌─────────────────────────────────────────────┐
│ 13. ALMACENAMIENTO EN BD                    │
│     • PostgreSQL: detección + metadata      │
│     • MongoDB: imagen + logs                │
│     • S3/Blob: imagen original + procesada  │
└─────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 14. ANALÍTICA Y REPORTES                    │
│     • Tasa de infestación                   │
│     • Mapa de calor (hotspots)              │
│     • Predicción de brotes                  │
│     • Recomendaciones de control            │
└─────────────────────────────────────────────┘
                 │
                 ▼
             [FIN]
```

---

## 4. ARQUITECTURA DE MICROSERVICIOS DETALLADA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARQUITECTURA DE MICROSERVICIOS                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────┐
│   API Gateway     │  Port: 80/443
│    (Kong/NGINX)   │
└─────────┬─────────┘
          │
          ├──────────────────┬──────────────────┬──────────────────┬──────────┐
          │                  │                  │                  │          │
          ▼                  ▼                  ▼                  ▼          ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌─────────┐ ┌────────────┐
│ Auth Service     │ │Detection Service │ │Environment Svc   │ │Alert Svc│ │Analytics   │
│                  │ │                  │ │                  │ │         │ │Service     │
│ • Login/Register │ │ • Image Upload   │ │ • Sensor Data    │ │• Notify │ │            │
│ • JWT Tokens     │ │ • Model Infer    │ │ • Weather API    │ │• Email  │ │• Reports   │
│ • RBAC           │ │ • Batch Process  │ │ • Multimodal     │ │• SMS    │ │• Dashboard │
│ • User Mgmt      │ │ • Tracking       │ │   Fusion         │ │• Push   │ │• Predict   │
│                  │ │                  │ │                  │ │         │ │            │
│ FastAPI/Python   │ │ FastAPI/Python   │ │ Node.js/Express  │ │Node.js  │ │Python      │
│ Port: 5001       │ │ Port: 5002       │ │ Port: 5003       │ │Port:5004│ │Port: 5005  │
└──────────────────┘ └──────────────────┘ └──────────────────┘ └─────────┘ └────────────┘
          │                  │                  │                  │          │
          │                  │                  │                  │          │
          └──────────────────┴──────────────────┴──────────────────┴──────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │      Message Broker          │
                        │      (RabbitMQ/Kafka)        │
                        │                              │
                        │  Topics:                     │
                        │  • image.uploaded            │
                        │  • detection.completed       │
                        │  • alert.triggered           │
                        │  • report.generated          │
                        └──────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  PostgreSQL 16   │       │   MongoDB 7      │       │   Redis 7        │
│                  │       │                  │       │                  │
│ Tables:          │       │ Collections:     │       │ Keys:            │
│ • users          │       │ • images         │       │ • session:*      │
│ • detections     │       │ • logs           │       │ • queue:*        │
│ • alerts         │       │ • metadata       │       │ • cache:*        │
│ • reports        │       │ • annotations    │       │ • pubsub:*       │
│ • sensors_data   │       │                  │       │                  │
└──────────────────┘       └──────────────────┘       └──────────────────┘
          │                            │                            │
          └────────────────────────────┴────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   Object Storage (S3/Blob)   │
                        │                              │
                        │  Buckets:                    │
                        │  • raw-images/               │
                        │  • processed-images/         │
                        │  • models/                   │
                        │  • reports/                  │
                        └──────────────────────────────┘
```

---

## 5. MODELO DE DEEP LEARNING - ARQUITECTURA DETALLADA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           ARQUITECTURA YOLOv8 + CBAM PARA DETECCIÓN DE H. virescens         │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT IMAGE: 640 x 640 x 3 (RGB)
    │
    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                         BACKBONE - CSPDarknet53                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Conv2D (32 filters, 6x6, stride=2) → 320x320x32               │  │
│  │  ↓                                                               │  │
│  │  C3 Block 1 (64 filters) → 160x160x64                          │  │
│  │  ↓                                                               │  │
│  │  C3 Block 2 (128 filters) → 80x80x128        [P3 Output]       │  │
│  │  ↓                                                   │           │  │
│  │  C3 Block 3 (256 filters) → 40x40x256        [P4 Output]       │  │
│  │  ↓                                                   │           │  │
│  │  C3 Block 4 (512 filters) → 20x20x512        [P5 Output]       │  │
│  │  ↓                                                   │           │  │
│  │  SPPF (Spatial Pyramid Pooling Fast)                │           │  │
│  │                                                       │           │  │
│  │  CBAM Attention Module aplicado en cada nivel        │           │  │
│  │  • Channel Attention (Global pooling + MLP)          │           │  │
│  │  • Spatial Attention (Conv + Sigmoid)                │           │  │
│  └─────────────────────────────────────────────────────┘           │  │
│                                                                      │  │
└──────────────────────────────────────────────────────────────────────┘
         │                       │                       │
         │ P3 (80x80x128)       │ P4 (40x40x256)       │ P5 (20x20x512)
         │                       │                       │
         ▼                       ▼                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    NECK - PANet + BiFPN                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Bidirectional Feature Pyramid Network (BiFPN)                  │  │
│  │                                                                  │  │
│  │  Top-down pathway (agregación de características):              │  │
│  │  P5 ─────→ Upsample + Concat ─→ P4'                            │  │
│  │  P4' ────→ Upsample + Concat ─→ P3'                            │  │
│  │                                                                  │  │
│  │  Bottom-up pathway (refinamiento):                              │  │
│  │  P3' ────→ Downsample + Add ──→ P4''                           │  │
│  │  P4'' ───→ Downsample + Add ──→ P5''                           │  │
│  │                                                                  │  │
│  │  Weighted Feature Fusion (learnable weights)                    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
         │                       │                       │
         │ P3'' (80x80)         │ P4'' (40x40)         │ P5'' (20x20)
         │                       │                       │
         ▼                       ▼                       ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    DETECTION HEAD (3 escalas)                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Para cada escala (P3, P4, P5):                                 │  │
│  │                                                                  │  │
│  │  1. Clasificación Head:                                         │  │
│  │     Conv2D → BatchNorm → SiLU → Conv2D                         │  │
│  │     Output: (B, 80, H, W, num_classes)                         │  │
│  │     • Softmax para probabilidades de clase                      │  │
│  │                                                                  │  │
│  │  2. Regresión Head (Bounding Box):                              │  │
│  │     Conv2D → BatchNorm → SiLU → Conv2D                         │  │
│  │     Output: (B, 80, H, W, 4)  # [x, y, w, h]                   │  │
│  │     • Coordenadas normalizadas                                  │  │
│  │                                                                  │  │
│  │  3. Objectness Head:                                            │  │
│  │     Conv2D → BatchNorm → SiLU → Conv2D                         │  │
│  │     Output: (B, 80, H, W, 1)                                   │  │
│  │     • Sigmoid para confianza de objeto                          │  │
│  │                                                                  │  │
│  │  Anchor-free approach (distribución gaussiana)                  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                  ┌─────────────────────────────┐
                  │    POST-PROCESSING          │
                  │                             │
                  │  1. NMS (IoU = 0.45)        │
                  │  2. Score threshold (0.7)   │
                  │  3. Class filtering         │
                  │  4. DeepSORT tracking       │
                  └─────────────────────────────┘
                                │
                                ▼
OUTPUT: List[(x1, y1, x2, y2, confidence, class_id, track_id)]
        └─ Bounding boxes con tracking ID


┌─────────────────────────────────────────────────────────────────────────┐
│                      MÓDULO DE FUSIÓN MULTIMODAL                         │
└─────────────────────────────────────────────────────────────────────────┘

Features CNN (1024-dim)              Datos Ambientales (5-dim)
       │                                     │
       │                                     │
       │  ┌──────────────────────┐          │
       │  │  Global Avg Pool     │          │
       │  │  Feature Extractor   │          │
       │  └──────────────────────┘          │
       │              │                      │
       └──────────────┴──────────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │   Concatenation     │
           │   (1029 features)   │
           └─────────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │   MLP Layer 1       │
           │   1029 → 512        │
           │   ReLU + Dropout    │
           └─────────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │   MLP Layer 2       │
           │   512 → 256         │
           │   ReLU + Dropout    │
           └─────────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │   Output Layer      │
           │   256 → 2           │
           │   Softmax           │
           └─────────────────────┘
                      │
                      ▼
         [Negativo, Positivo] + Confianza ajustada

Entradas ambientales:
1. Temperatura (°C)
2. Humedad relativa (%)
3. Fenología (0-5: codificado)
4. Hora del día (0-23)
5. Día juliano (1-365)
```

---

## 6. STACK TECNOLÓGICO COMPLETO

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STACK TECNOLÓGICO 2025                              │
└─────────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════╗
║                        FRONTEND & INTERFAZ                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  • React 18.3+ con TypeScript 5.3+                                       ║
║  • Next.js 14+ para SSR y optimización                                   ║
║  • Material-UI (MUI) v5 / Tailwind CSS 3.4+                              ║
║  • Recharts / Chart.js para visualización                                ║
║  • Leaflet / Mapbox GL JS para mapas                                     ║
║  • React Query para estado servidor                                      ║
║  • Zustand/Redux Toolkit para estado global                              ║
║  • PWA con Service Workers (offline-first)                               ║
║  • Vite como build tool                                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                        BACKEND & API                                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Microservicio de Inferencia:                                            ║
║  • FastAPI 0.115+ (Python 3.11+)                                         ║
║  • Uvicorn ASGI server                                                   ║
║  • Pydantic v2 para validación                                           ║
║                                                                           ║
║  Microservicio de Notificaciones:                                        ║
║  • Node.js 20 LTS + Express.js 4.18+                                     ║
║  • Socket.IO para WebSocket                                              ║
║  • Bull queue con Redis                                                  ║
║                                                                           ║
║  (Opcional) Backend Empresarial:                                         ║
║  • Spring Boot 3.2+ (Java 21)                                            ║
║  • Spring Data JPA                                                       ║
║  • Spring Security                                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                    DEEP LEARNING & AI                                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Framework Principal:                                                     ║
║  • PyTorch 2.5+ con CUDA 12.1                                            ║
║  • Ultralytics YOLOv8/v9                                                 ║
║  • TorchVision 0.20+                                                     ║
║                                                                           ║
║  Preprocesamiento:                                                        ║
║  • OpenCV 4.10+                                                          ║
║  • Albumentations 1.4+                                                   ║
║  • Pillow 10.4+                                                          ║
║                                                                           ║
║  Inferencia Optimizada:                                                  ║
║  • ONNX Runtime 1.19+                                                    ║
║  • TensorFlow Lite 2.17+ (para edge)                                     ║
║  • OpenVINO (Intel optimization)                                         ║
║                                                                           ║
║  Tracking:                                                               ║
║  • DeepSORT / ByteTrack                                                  ║
║                                                                           ║
║  Experimentación:                                                        ║
║  • Weights & Biases (W&B)                                                ║
║  • MLflow 2.16+                                                          ║
║  • TensorBoard                                                           ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                         BASES DE DATOS                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  SQL (Relacional):                                                        ║
║  • PostgreSQL 16.4 con PostGIS                                           ║
║  • Extensiones: TimescaleDB (series temporales)                          ║
║                                                                           ║
║  NoSQL (Documentos):                                                      ║
║  • MongoDB 7.0+ (Community/Atlas)                                        ║
║                                                                           ║
║  Cache & Queue:                                                          ║
║  • Redis 7.4+ (in-memory)                                                ║
║  • Redis Stack (RedisJSON, RedisSearch)                                  ║
║                                                                           ║
║  Object Storage:                                                         ║
║  • AWS S3 / Azure Blob / Google Cloud Storage                            ║
║  • MinIO (alternativa self-hosted)                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                    MESSAGE BROKER & STREAMING                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  • RabbitMQ 3.13+ (AMQP)                                                 ║
║  • Apache Kafka 3.8+ (alternativa para high-throughput)                  ║
║  • MQTT (Mosquitto 2.0+) para IoT                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                    CONTAINERIZACIÓN & ORCHESTRATION                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  • Docker 27+                                                            ║
║  • Docker Compose v2                                                     ║
║  • Kubernetes 1.31+                                                      ║
║    - EKS (AWS) / AKS (Azure) / GKE (Google)                              ║
║  • Helm 3.16+ (package manager)                                          ║
║  • Istio / Linkerd (service mesh, opcional)                              ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                         CLOUD PROVIDERS                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  AWS (Amazon Web Services):                                              ║
║  • EC2 (GPU: g4dn.xlarge con NVIDIA T4)                                  ║
║  • Lambda (serverless)                                                   ║
║  • S3 (storage)                                                          ║
║  • RDS for PostgreSQL                                                    ║
║  • SageMaker (ML training, opcional)                                     ║
║  • CloudWatch (monitoring)                                               ║
║                                                                           ║
║  Azure (Microsoft):                                                      ║
║  • Azure VM (NCSv3 con Tesla V100)                                       ║
║  • Azure Functions                                                       ║
║  • Blob Storage                                                          ║
║  • Azure Database for PostgreSQL                                         ║
║  • Azure ML                                                              ║
║  • Azure Monitor                                                         ║
║                                                                           ║
║  GCP (Google Cloud Platform):                                            ║
║  • Compute Engine (GPU: n1-standard con T4)                              ║
║  • Cloud Run                                                             ║
║  • Cloud Storage                                                         ║
║  • Cloud SQL                                                             ║
║  • Vertex AI                                                             ║
║  • Cloud Monitoring                                                      ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                         EDGE COMPUTING                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Hardware:                                                               ║
║  • NVIDIA Jetson Nano 4GB / Xavier NX                                    ║
║  • Raspberry Pi 4 Model B (8GB RAM)                                      ║
║  • ESP32-CAM (AI-Thinker)                                                ║
║                                                                           ║
║  Software Edge:                                                          ║
║  • JetPack SDK 5.1+ (Jetson)                                             ║
║  • Raspberry Pi OS 64-bit                                                ║
║  • TensorFlow Lite / ONNX Runtime                                        ║
║  • Edge Impulse (opcional)                                               ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                         IoT & SENSORES                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Sensores Ambientales:                                                   ║
║  • DHT22 (temperatura & humedad)                                         ║
║  • BME280 (temperatura, humedad, presión)                                ║
║  • Soil moisture sensor                                                  ║
║                                                                           ║
║  Cámaras:                                                                ║
║  • Raspberry Pi HQ Camera (12.3MP)                                       ║
║  • ESP32-CAM (2MP)                                                       ║
║  • Arducam (8MP autofocus)                                               ║
║                                                                           ║
║  Protocolo:                                                              ║
║  • MQTT (Mosquitto broker)                                               ║
║  • HTTP REST API                                                         ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                    MONITORING & OBSERVABILITY                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Métricas:                                                               ║
║  • Prometheus 2.54+                                                      ║
║  • Grafana 11+                                                           ║
║                                                                           ║
║  Logs:                                                                   ║
║  • ELK Stack (Elasticsearch, Logstash, Kibana) 8.15+                     ║
║  • Fluentd / Fluent Bit                                                  ║
║                                                                           ║
║  Tracing:                                                                ║
║  • Jaeger / Zipkin                                                       ║
║  • OpenTelemetry                                                         ║
║                                                                           ║
║  APM:                                                                    ║
║  • Datadog / New Relic (comercial)                                       ║
║  • Sentry (error tracking)                                               ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                         CI/CD & DevOps                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  • GitHub Actions / GitLab CI                                            ║
║  • ArgoCD (GitOps para Kubernetes)                                       ║
║  • Terraform para IaC                                                    ║
║  • Ansible para configuration management                                 ║
║  • SonarQube para code quality                                           ║
║  • Trivy para security scanning                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                         SEGURIDAD                                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  • OAuth 2.0 + OpenID Connect                                            ║
║  • JWT (JSON Web Tokens)                                                 ║
║  • bcrypt / Argon2 (password hashing)                                    ║
║  • SSL/TLS con Let's Encrypt                                             ║
║  • AWS WAF / Cloudflare                                                  ║
║  • HashiCorp Vault (secrets management)                                  ║
║  • OWASP ZAP (pentesting)                                                ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 7. DIAGRAMA DE DESPLIEGUE EN CLOUD (AWS)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ARQUITECTURA DE DESPLIEGUE - AWS                        │
└─────────────────────────────────────────────────────────────────────────────┘

                          ┌──────────────────────┐
                          │   Route 53 (DNS)     │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │   CloudFront (CDN)   │
                          │   + SSL Certificate  │
                          └──────────┬───────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VPC (Virtual Private Cloud)                     │
│  Region: us-east-1 (N. Virginia)  CIDR: 10.0.0.0/16                         │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          PUBLIC SUBNET (10.0.1.0/24)                   │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Application Load Balancer (ALB)                                 │  │ │
│  │  │  • HTTPS listener (443)                                          │  │ │
│  │  │  • Target Groups: Frontend, API                                  │  │ │
│  │  │  • Health Checks                                                 │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  NAT Gateway                                                     │  │ │
│  │  │  • Elastic IP                                                    │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                         │
│                 ┌───────────────────┼───────────────────┐                    │
│                 │                   │                   │                    │
│  ┌──────────────▼───────────────────────────────────────▼──────────────┐   │
│  │              PRIVATE SUBNET - APP (10.0.2.0/24)                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  EKS Cluster (Elastic Kubernetes Service)                   │    │   │
│  │  │  Kubernetes v1.31                                            │    │   │
│  │  │                                                               │    │   │
│  │  │  ┌──────────────────┐  ┌──────────────────┐                 │    │   │
│  │  │  │  Frontend Pods   │  │  Backend Pods    │                 │    │   │
│  │  │  │  (React)         │  │  (FastAPI)       │                 │    │   │
│  │  │  │  Replicas: 3     │  │  Replicas: 5     │                 │    │   │
│  │  │  │  CPU: 1 vCPU     │  │  CPU: 2 vCPU     │                 │    │   │
│  │  │  │  RAM: 2GB        │  │  RAM: 4GB        │                 │    │   │
│  │  │  └──────────────────┘  └──────────────────┘                 │    │   │
│  │  │                                                               │    │   │
│  │  │  ┌──────────────────┐  ┌──────────────────┐                 │    │   │
│  │  │  │  Node.js Pods    │  │  Alert Pods      │                 │    │   │
│  │  │  │  (WebSocket)     │  │  (Notificación)  │                 │    │   │
│  │  │  │  Replicas: 3     │  │  Replicas: 2     │                 │    │   │
│  │  │  └──────────────────┘  └──────────────────┘                 │    │   │
│  │  │                                                               │    │   │
│  │  │  Node Groups: t3.large (On-Demand) + t3.large (Spot)        │    │   │
│  │  │  Auto Scaling: Min=3, Desired=5, Max=10                     │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  EC2 Instances para ML Inference (GPU)                      │    │   │
│  │  │  Instance Type: g4dn.xlarge (NVIDIA T4 16GB)               │    │   │
│  │  │  • PyTorch inference server                                 │    │   │
│  │  │  • Model: YOLOv8 + CBAM                                     │    │   │
│  │  │  • Auto Scaling: Min=1, Max=5                               │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                     │                                         │
│  ┌──────────────────────────────────▼───────────────────────────────────┐   │
│  │              PRIVATE SUBNET - DATA (10.0.3.0/24)                      │   │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────┐ │   │
│  │  │  RDS PostgreSQL    │  │  ElastiCache Redis │  │  DocumentDB    │ │   │
│  │  │  (Multi-AZ)        │  │  (Cluster Mode)    │  │  (MongoDB API) │ │   │
│  │  │  db.r6g.large      │  │  cache.r6g.large   │  │  db.r6g.large  │ │   │
│  │  │  Storage: 500GB    │  │  Nodes: 3          │  │  Storage:500GB │ │   │
│  │  │  Backup: Daily     │  │  RAM: 16GB/node    │  │  Backup: Daily │ │   │
│  │  └────────────────────┘  └────────────────────┘  └────────────────┘ │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVICIOS ADICIONALES AWS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  S3 Buckets      │  │  Lambda          │  │  SQS/SNS         │         │
│  │                  │  │                  │  │                  │         │
│  │ • raw-images     │  │ • Image resize   │  │ • Message Queue  │         │
│  │ • processed      │  │ • Thumbnail gen  │  │ • Push notif     │         │
│  │ • models         │  │ • PDF reports    │  │                  │         │
│  │ • backups        │  │                  │  │                  │         │
│  │                  │  │ Trigger: S3      │  │                  │         │
│  │ Lifecycle:       │  │ Memory: 3GB      │  │                  │         │
│  │ IA after 30d     │  │ Timeout: 5min    │  │                  │         │
│  │ Glacier after 90d│  │                  │  │                  │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  CloudWatch      │  │  X-Ray           │  │  Secrets Manager │         │
│  │  • Logs          │  │  • Tracing       │  │  • DB passwords  │         │
│  │  • Metrics       │  │  • Performance   │  │  • API keys      │         │
│  │  • Alarms        │  │                  │  │  • JWT secrets   │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  IAM             │  │  WAF             │  │  Shield          │         │
│  │  • Roles         │  │  • Rate limiting │  │  • DDoS protect  │         │
│  │  • Policies      │  │  • Geo blocking  │  │                  │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         MONITOREO Y DISASTER RECOVERY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Prometheus + Grafana (EKS)                                                │
│  • ELK Stack (Elasticsearch, Logstash, Kibana) en EC2                        │
│  • Backup automático diario a S3                                             │
│  • Cross-region replication (us-west-2 como DR)                              │
│  • RTO: 1 hora, RPO: 15 minutos                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              EDGE LOCATIONS                                  │
│  (Virú, La Libertad - On Premises)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  NVIDIA Jetson Xavier NX                                             │  │
│  │  • Model: YOLOv8-lite (ONNX)                                         │  │
│  │  • Local inference: 20-25 FPS                                        │  │
│  │  • Sync to cloud every 1 hour                                        │  │
│  │  • 4G/5G connectivity                                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Raspberry Pi 4 (8GB) + HQ Camera                                    │  │
│  │  • Image capture: 12MP                                               │  │
│  │  • Send to Jetson for inference                                      │  │
│  │  • MQTT protocol                                                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Sensores IoT (DHT22, BME280)                                        │  │
│  │  • Readings every 10 minutes                                         │  │
│  │  • MQTT broker local                                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

Este documento proporciona una arquitectura completa y detallada basada en las mejores prácticas y tecnologías actuales de 2025. ¿Necesitas que desarrolle alguna sección adicional o genere diagramas visuales en formato de imagen?
