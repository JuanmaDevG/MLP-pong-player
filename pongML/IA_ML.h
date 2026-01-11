// Activa la IA por Machine Learning
#define MACHINE_LEARNING

// Datos recabados por el juego durante la ejecución
// (Siempre referidos a la pala de la izquierda)
typedef enum {
   s_jugadorVY = 0,     // Velocidad Y
   s_enemigoYRel,       // Posición Y de la pala contraria (Relativa a nuestra posición)
   s_enemigoVY,         // Velocidad Y de la pala contraria
   s_pelotaXRel,        // [ (X,Y) de la pelota relativas a nuestra X e Y
   s_pelotaYRel,        // ]
   s_pelotaVX,          // [ (Vx, Vy)  de la pelota
   s_pelotaVY,          // ]
   s_teclaArriba,       // Tecla arriba pulsada (1, 0)
   s_teclaAbajo,        // Tecla abajo pulsada (1, 0)
   s_N_EstadosJuego     // Número total de características
} SEstadoJuego;

extern double vEstadoJuego[s_N_EstadosJuego];