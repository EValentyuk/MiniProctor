# Черновики поста о MiniProctor

Готовы к копированию. Перед публикацией -- замени ссылку на демо-видео, если запишешь его и зальёшь, например, на YouTube или TenChat.

---

## Русский -- развёрнутый (TenChat, LinkedIn для русскоязычной аудитории)

> За неделю собрал пет-проект MiniProctor -- MVP-системы прокторинга на компьютерном зрении. Цель -- показать команде, которая делает приёмку экзаменов в вузах, что я умею не только писать ТЗ, но и доводить ML-проект до работающего демо.
>
> Что внутри:
> -- 5 детекторов нарушений: положение головы и взгляд (MediaPipe), несколько лиц в кадре, отсутствие лица, телефон и книга (YOLOv8 fine-tuned);
> -- покадровая оценка precision / recall / F1 на 7 размеченных тестовых клипах;
> -- интерактивный Streamlit UI с таймлайном событий и экспортом CSV;
> -- C4-диаграммы архитектуры, портфельный отчёт с привязкой каждого технического решения к навыкам системного анализа.
>
> Главный сюжет -- fine-tune YOLO. Pretrained на COCO давал по книгам F1 всего 0.07. После fine-tune только на чужих Roboflow-данных -- регрессия из-за доменного сдвига. После добавления 89 моих собственных кадров через open-vocabulary разметку YOLO-World -- F1 вырос в 5 раз, по телефону -- на четверть.
>
> Код, документация и метрики открыты: https://github.com/EValentyuk/MiniProctor
>
> Если вы из ML-команды, которая делает прокторинг или образовательные продукты -- буду рад поговорить.
>
> #ML #computervision #yolo #mediapipe #прокторинг #portfolio #системныйанализ

---

## English -- expanded (LinkedIn for international audience)

> Built MiniProctor in a week -- a CV-based proctoring MVP. Goal: show ML teams in EdTech that I, as a systems analyst transitioning to ML, can ship a complete project end-to-end, not just write requirements docs.
>
> What's inside:
> -- 5 violation detectors: head pose / gaze (MediaPipe), multi-face, no-face, phone and book (fine-tuned YOLOv8);
> -- per-frame precision / recall / F1 evaluation on 7 labeled test clips;
> -- interactive Streamlit UI with an event timeline and CSV export;
> -- C4 architecture diagrams, portfolio report linking each technical decision to systems analysis skills.
>
> The main technical story is the YOLO fine-tune. Pretrained on COCO gave only F1 = 0.07 for books. Fine-tune on Roboflow data alone caused a regression (classic domain shift). Adding 89 of my own webcam frames -- auto-labeled by YOLO-World as the open-vocabulary detector -- lifted book F1 by 5x and phone F1 by a quarter.
>
> Code, docs and metrics are open: https://github.com/EValentyuk/MiniProctor
>
> If you're in an EdTech / proctoring ML team -- happy to chat.
>
> #ML #computervision #yolo #mediapipe #proctoring #portfolio #systemsanalysis

---

## Короткий вариант (Twitter / X, до 280 символов)

> Pet-проект за неделю: MiniProctor, MVP прокторинга на CV. 5 детекторов, YOLO fine-tune, Streamlit UI с таймлайном. Book F1 0.07 → 0.37 за счёт auto-labeling через YOLO-World.
>
> https://github.com/EValentyuk/MiniProctor

(Английская версия 280 символов:)

> 1-week pet project: MiniProctor, a CV proctoring MVP. 5 detectors, YOLO fine-tune, Streamlit UI with event timeline. Book F1 jumped 0.07 → 0.37 via YOLO-World auto-labeling.
>
> https://github.com/EValentyuk/MiniProctor

---

## Заметки по публикации

- **Время постинга.** В LinkedIn оптимально вторник-четверг, 10:00-11:00 по местному времени аудитории. В TenChat -- утром (8-10) или вечером (18-20).
- **Картинка.** К посту обязательно прикрепить картинку. Самый сильный вариант -- скриншот таймлайна из Streamlit с подписями нарушений. Альтернатива -- любая из C4-диаграмм из `docs/diagrams/images/`.
- **Демо-видео.** Если запишете -- залейте в YouTube unlisted, дайте ссылку в посте отдельной строкой. В TenChat можно прикрепить видео напрямую.
- **Хэштеги.** В LinkedIn 3-5 хэштегов работают лучше, чем 10. В TenChat хэштеги менее критичны, можно меньше.
- **Ссылка.** Не первой строкой -- LinkedIn-алгоритм снижает охват постов со ссылкой в начале. Ссылку -- в конце, после содержания.
- **CTA.** В конце поста -- прямой призыв «Если вы из такой-то команды, напишите». Без него люди читают и забывают.
