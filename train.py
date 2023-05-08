from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import DefaultDataCollator
import pandas as pd
from datasets import load_dataset
import ast
import os
from huggingface_hub import login
def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]
    pad_on_right = tokenizer.padding_side == "right"
    max_length = 384 # The maximum length of a feature (question and context)
    doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = 0 #input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        answers = ast.literal_eval(answers)
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples
def preprocess_function(examples):

    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(

        questions,

        examples["context"],

        max_length=500,

        truncation="only_second",

        return_offsets_mapping=True,

        padding="max_length",

    )

    offset_mapping = inputs.pop("offset_mapping")

    answers = examples["answers"]

    start_positions = []

    end_positions = []

    for i, offset in enumerate(offset_mapping):

        answer = answers[i]
        answer = ast.literal_eval(answer)
        start_char = answer["answer_start"][0]

        end_char = answer["answer_start"][0] + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context

        idx = 0

        while sequence_ids[idx] != 1:

            idx += 1

        context_start = idx

        while sequence_ids[idx] == 1:

            idx += 1

        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:

            start_positions.append(0)

            end_positions.append(0)

        else:

            # Otherwise it's the start and end token positions

            idx = context_start

            while idx <= context_end and offset[idx][0] <= start_char:

                idx += 1

            start_positions.append(idx - 1)

            idx = context_end

            while idx >= context_start and offset[idx][1] >= end_char:

                idx -= 1

            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions

    inputs["end_positions"] = end_positions

    return inputs
login(token="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")
# Load the pre-trained model
model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained("C:/Users/nurgaliy/.cache/huggingface/hub/models--bigscience--bloom-1b1/snapshots/6f4195539db0eef1c9d010289f32e0645d9a2354")
#model = AutoModelForQuestionAnswering.from_pretrained("C:/Users/nurgaliy/.cache/huggingface/hub/models--bigscience--bloom-1b1/snapshots/6f4195539db0eef1c9d010289f32e0645d9a2354")
dataset = [{"question": "How much does a PhD cost?", "answer": "Current fees are around 260 euro per year, of which 210 euro are for academic tutoring and the remainder for admin  fees and medical insurance in the event of a school-related accident. For more information, please see the following section: 'Doctoral School > Useful Information > Regulations > Decree on Public Fees'."}, {"question": "Requesting and Obtaining Your Master's or PhD Degree Certificate", "answer": "If you reside in Majorca, Minorca or Ibiza: In order to request and obtain your master's degree certificate, you need to go to the administrative services in the Centre for Postgraduate Studies (CEP) and the Doctoral School (EDUIB) in-person, in the Antoni M. Alcover i Sureda building or the Minorca and Ibiza headquarters. You will need to bring a valid original ID card, NIE or passport as well as a photocopy.You must book an appointment in the 'Appointment' app at the following link: Centre for Postgraduate Studies > How can we help? > Appointments or in Doctoral School > Appointments. You will need to review and sign a form before we authorise payment of the degree certificate issuance fee. A fee must be paid for the degree certificate to be issued. The fee is set by the Decree on public fees for academic and administrative services. You may pay by credit or debit card at admin services or make a cash deposit directly into the UIB's account. Students will be given a certificate slip after performing this procedure. The slip has the same validity as the official degree certificate until the final certificate is issued by the Student and Degrees Service (SAT). Once the final degree certificate is issued, the SAT will send a message to the student's e-mail address as listed in UIBdigital with a collection notification.For more information, please see: Student and Degrees Service (SAT), in Son Lledó. N.B.: In order to obtain a degree transcript, you need to pay a fee of 17.91 euro (8.96 euro for large families). If you do NOT reside in Majorca, Minorca or Ibiza: You must send an e-mail to postgrauarrobauib.es, stating your name, ID number and an issuance request for your master's degree certificate. You will receive a reply setting out the steps you must follow for the procedure. The e-mail will contain a request receipt as an attachment: if the personal details in the request are correct, you must reply to the e-mail with your request correctly filled out, signed, and scanned, in addition to a scanned copy of your valid ID card, NIE or passport. If any personal details are missing (address, phone number, postcode, etc.), you may add them by hand or, where necessary, make any amendments. N.B.: your ID number will appear without the final letter. This is not a mistake and should, therefore, NOT be amended. You will need to pay the fee set out in the Decree on public fees for academic and administrative services into the UIB's account, stating your name in the item field. You will need to submit a scanned copy of the payment receipt. Once you have completed these steps, we will send a degree slip to your home address. This slip has the same validity as an official degree certificate until the final certificate is issued by the SAT. UIB payment account: Bank: CaixaBank, SA IBAN: ES27 2100 7359 7113 0010 4267 BIC: CAHMESMM N.B.: In order to obtain a degree transcript, you will need to pay a separate fee of 17.91 euro (8.96 euro for large families). This fee must be paid into the same account (payment for the degree certificate and degree transcript needs to be done separately since they are two different items) via bank transfer or direct cash deposit into the UIB's account. Once the payment is made, you will need to scan and attach the payment receipt (deposit/transfer slip, etc.)."}, {"question": "Where can I find information about grants?", "answer": "You may find information about grants at: https://estudis.uib.cat/informacioperalumnes/Beques-i-ajuts/, https://edoctorat.uib.cat/Informacio/Beques/"}, {"question": "I want to do a PhD: how do I go about it?", "answer": "The first step is to pre-register. Pre-registration is done online by clicking 'Pre-registration' on the degree you are interested in on the 'On offer' page or via the app: <https://postgrau.uib.es/>. You need to upload the following documentation to the app: ID,, Curriculum, Qualifications that provide admission to the PhD (bachelors, masters, etc.): official academic qualification (or the receipt of having requested it) and transcript. If the qualifications are from the UIB, there is no need to upload this documentation. If your admission qualification is from overseas, please see the 'Students with foreign degrees' section on our website. You may view the pre-registration and registration deadlines in the 'Pre-registration and admissions - Calendars' section on this website."}, {"question": "What original documentation do I need to submit? When and how do I do it?", "answer": "Once registered, we will notify you of the deadline and how to submit or send admissions documentation (ID, qualification, official academic transcript...) to the Doctoral School. You will also receive information about the procedures you need to do during the first year (Producing the research plan, your CV in GREC, signing the thesis charter…). You may view more information in 'Instructions' on the pre-registration and registration page, and in the 'Procedures' section."}, {"question": "Which qualifications provide admission to a PhD?", "answer": "Qualifications that provide admission to PhD programmes are regulated by Article 6 of Royal Decree 99/2011, of 28th January, which governs official PhD studies. You may view it at BOE 35, of 10/02/2011 - Royal Decree 99/2011, of 28th January, which governs official PhD studies. You may also view the UIB's own regulations for PhD studies in the 'UIB Doctoral School Regulations' section."}, {"question": "Do I need to take face-to-face classes?", "answer": "On a PhD programme, there are no classes per se, although depending on the programme where you are registered, there may be some face-to-face activities.There may be two types of activities: Cross-cutting: the academic commissions for each PhD programme decide on whether these activities are obligatory or elective for students. They are organised by the Doctoral School and are common to all PhD programmes. For more information, please see the Doctoral School Training Activities. Specific: the academic commissions for each PhD programme decide on whether these activities are obligatory or elective, and whether they are specific to their students. For more information about these activities, please see the PhD studies on offer section. You will need to select your PhD programme and go to the 'Training Activities' section. If you have any academic questions, you may contact the programme coordinator via the contact form in the 'General Information' section for the programme."}, {"question": "How are thesis tutors and supervisors assigned?", "answer": "Both the thesis tutor and supervisor are assigned by the academic commission. Your pre-registration application must include a summary of your research project and area. You will also need to make your own thesis supervisor proposal, which will be taken into account when assigning your thesis supervisor(s)."}, {"question": "Can I change my assigned thesis supervisor?", "answer": "Yes, your thesis supervisor can be changed. In order to request a change, you need to fill out the form via the <https://postgrau.uib.es> platform and attached the withdrawal document signed by the thesis supervisor(s) and an acceptance document signed by the new supervisor(s). The corresponding academic commission for the PhD programme must approve the change. You may view further information in the Doctoral School Procedures section."}, {"question": "Who grants admission to a PhD programme?", "answer": "Admissions to each PhD programme are approved by their corresponding academic commission. Admissions are subject to the criteria set out in the curricula, which are available on the institutional website on the pages corresponding to each programme (you may view the 'On offer' section to see the requirements for the programme you are interested in). For more general information on the pre-registration and admissions process, please see the 'Pre-registration and admissions - Instructions' section."}, {"question": "When do I need to register?", "answer": "Once admitted, you need to perform self-registration within the stated deadline via UIBdigital. When you are in UIBdigital, you need to click the Training - Registration - Self-registration tabs, click on the 'pencil' and, after accepting the general terms and conditions, you may formalise your registration. UIBdigital credentials are the same as those you were given during pre-registration. You may view more information on registration in the 'Initial registration' section. You may view the pre-registration and registration deadlines in the 'Pre-registration and admissions - Calendars' section on this website."}, {"question": "What does a PhD comprise?", "answer": "Candidates undertake and produce original research in any field of knowledge: PhD thesis. Doctorands must pass annual assessments throughout their enrolment period (initial and subsequent assessments) that serve to monitor their research."}, {"question": "How long does the PhD last?", "answer": "By default, all PhD students are full-time, except for those who request part-time student status for justified reasons: work, dependents, etc. Full-time: three years, with the possibility of a one-year extension that may, exceptionally, be extended a further year. Part-time: five years, with the possibility of a two-year extension that may, exceptionally, be extended a further year. In order to request part-time student status, you must have formalised your PhD registration. (More information.) Temporary leave: this can be requested for a maximum of one year, which may be extended a further year. After the temporary leave period ends, students must request to re-join the PhD and register for the corresponding academic year. (More information.) Extensions, temporary leave and part-time student status must be requested from the Academic Commission over the <https://postgrau.uib.es> platform. For further information please see the Doctoral School attendance requirements section."}, {"question": "Can I spend a year without registering?", "answer": "According to Article 17.1 of Regulatory Agreement 13084, of 10th April 2019 (FOU 481): 'In the event of not registering during the relevant period for academic tutoring, doctorands may not register on the same PhD programme for the following two years. In any event, the academic commission for the relevant PhD programme must approve admission and the same academic file shall be maintained. This readmission may only be granted once.' For further information, please see the PhD readmission section in the Doctoral School Procedures section."},{"question": "Who can act as a thesis supervisor?", "answer": "The thesis supervisor must hold a Spanish or overseas PhD and have accredited experience in research, regardless of the university, centre or institution where s/he works."},{"question": "What grants are available for master's degrees?", "answer": "The Ministry of Education and Vocational Training offers general and mobility grants for university students every year. In addition, other institutions and organisations may occasionally offer grants and financial aid for postgraduate studies. The selection changes every year and, therefore, the UIB has set up a website for current information. You may also check the grants, awards and general financial aid section on the Centre for Postgraduate Studies website."}, {"question": "Is a photograph required?", "answer": "Students must upload a photograph to AppCrue in order to complete the personal details in their academic record and be identified on their online student ID card. For more information about the photo, please click here https://cep.uib.eu/en/Com_hi_puc_accedir/Matricula/Foto_targeta/."}, {"question": "What is the online student ID card and how do I get one?", "answer": "The online student ID card identifies you as a UIB student. Please bear in mind that in order to complete the process in the online services portal (like the online registry), you must have a digital certificate (certain procedures may be completed by using your UIB credentials). More information about the online student ID card here: https://tic.uib.cat/Servei/cataleg_serveis/Targeta-universitaria/."}, {"question": "What is the difference between an official and unofficial master's degree?", "answer": "Official master's degrees: their clear and explicit designation is 'Master's Degree in...'. These programmes aim for students to acquire advanced, specialised or multidisciplinary academic training geared towards an academic or professional specialisation, or to promote the start of research activity. These degrees are official and valid across Spain, with full academic recognition, and enable students to work in regulated professions in accordance with applicable regulations in each instance. Passing an official master's programme grants you the right to obtain the degree certificate with the specific RUCT designation (University and Degree Registry at the Ministry of Education, Culture and Sport). Unofficial master's degrees: these programmes may not use the designation 'Master's Degree in…'. They are included in the postgraduate programme selection, are UIB-specific and represent an additional academic option. These master's programmes and other UIB-specific courses (University Specialist/Expert) enable the university to provide an agile quality response to academic requirements, as well as refresher courses for students (for society in general and for employment opportunities). They are an essential element in our education system and provide top-flight and necessary training."}, {"question": "What are the enrolment requirements for master's degrees?", "answer": "Students with a Spanish degree or equivalent, a validated overseas degree or a degree granted equivalent status to Spanish undergraduate degrees who wish to enrol for a master's degree at the UIB must first complete a pre-enrolment process, where they can apply for one or more UIB master's programmes. All information about the procedure and required documents is available at the following link: https://cep.uib.eu/en/Com_hi_puc_accedir/Preinscripcio/. The website also provides information on how the Centre for Postgraduate Studies decides on candidates' eligibility for master's degrees and how the coordinators for each master's programme decide on admission to their specific degree. Students with overseas qualifications will find detailed relevant information at the following link: https://cep.uib.eu/en/Com_hi_puc_accedir/Titulacions_estrangeres/. Please find further information at: https://cep.uib.eu/en/Com_hi_puc_accedir/"}, {"question": "How long does a master's degree last?", "answer": "Master's degree curricula comprise between 60 and 120 credits. The academic arrangements for the programmes are split across two academic years. The total number of credits for each academic year has been established at 60 credits."}, {"question": "What is the minimum number of credits I can enrol for?", "answer": "Upon guidance from the university, the Government of the Balearic Islands establishes the minimum number of credits where students must enrol each academic year. Students embarking on a master's programme, or enrolling in a PhD training period regulated by Royal Decree 1393/2007 of 29th October (available in the 'Regulations -> General Regulations' section), must enrol for at least 30 credits. This number includes adapted or recognised credits, where applicable. Where students are granted part-time status, the amount drops to 15 credits (in accordance with relevant regulations). More information is available at the following link: 'Information on Academic Regulations, Academic Progress and Attendance Requirements and the Minimum Number of Initial Enrolled Credits'"}, {"question": "What is the minimum number of credits I must pass on a master's programme to continue studying at the UIB?", "answer": "The UIB has established a general academic progress plan, depending on whether students are full- or part-time: Full-time master's students must pass at least 40% of their enrolled credits (rounded up) for the academic year, Part-time master's students must pass at least 10% of their enrolled credits (rounded up) for the academic year, Where students only enrol for one subject, the aforementioned rules do not apply. More information is available at the following link: 'Information on Academic Regulations, Academic Progress and Attendance Requirements and the Minimum Number of Initial Enrolled Credits'."}, {"question": "Part-time student status", "answer": "Students who wish to take a master's programme part-time, and who fulfil the requirements, must submit their application to the administrative services in the Antoni Maria Alcover i Sureda building before formalising their enrolment. For more information on becoming a 'part-time student', please see the https://cep.uib.eu/en/normativa/#Reglament_Academic"}, {"question": "How can I get more information?", "answer": "On the Centre for Postgraduate Studies website https://cep.uib.eu/en/, By sending an e-mail to postgrau@uib.es, Or by making an appointment."}, {"question": "What is pre-registration?", "answer": "Pre-registration is the application that students submit in order to obtain a place on a master's degree programme at the UIB. Candidates may apply for admission to a maximum of three master's degree programmes. This application shall determine whether students meet the entry requirements for admission to a master's degree programme or a pathway/specialisation within a specific programme. Then, the relevant body of the master's degree programme shall review and assess the applications of those who meet the requirements and allocate the places. Depending on each master's programme https://cep.uib.cat/master/, admission may be conditional on: one or several pathways/specialisations on the master's programme, the requirement for specific prior training in some disciplines. For further information, please view the 'Admission criteria' section on the webpage for each master's degree programme, the need for students to take tests to ensure their level is suitable, admission tests to rank candidates, the need for supplementary training."}, {"question": "Who may participate?", "answer": "All individuals who meet one of the following entry requirements (or who can accredit them within the relevant amendment period): A Spanish undergraduate qualification or equivalent*, Degrees at the same level as Spanish undergraduate or master's qualifications, issued by universities or higher education institutions belonging to another member state within the European Higher Education Area (EHEA), enabling access to master's studies in that country, Overseas qualifications from education systems outside the EHEA which are equivalent to undergraduate degrees, without the need for official validation, but subject to verification of the standard of education of the degree, and that it enables access to postgraduate programmes in the country of issuance, Students who still need to pass their final degree project and, at most, 9 ECTS credits (conditional enrolment), only for those programmes for which it is so established in each call, and always prioritising the enrolment of those who hold an undergraduate degree or equivalent. * Undergraduate students who have not yet finished their degree programme, but who plan on passing their degree before the end of the deadline for amendments to pre-register for the requested master's programme, may fill in the pre-registration form and even click on 'Formalise application'. Nevertheless, the application will remain conditional on students finishing their degree programme and paying for their degree certificate to be issued before the end of the amendment period for each programme. They will need to attach their official certificate and transcript once the UIB notifies them of the official required documentation. The deadline to provide this documentation ends after the set period on each master's programme to submit amendments."}, {"question": "What is conditional enrolment?", "answer": "In accordance with RD 822/2021Document reading and Regulatory Agreement 14423/2022, students who still need to pass their final degree project and, at most, nine ECTS credits may formalise conditional enrolment for master's programmes where they have been admitted. These students will be subject to the same regulations as other enrolled students on the programme.Nevertheless, applications from students who already have their degree or equivalent will be prioritised. Conditional enrolment will only be possible for those programmes offering it in the application period. Moreover, students who formalise conditional enrolment will not be awarded their master's degree if they fail to pass their undergraduate degree programme. Candidates must check whether the master's programme offers conditional enrolment. Please check the available places 2022-23Document reading. Students who apply for conditional enrolment and an MEFP grant must ensure they fulfil the terms and conditions set out in the grant call."}, {"question": "Can students who have not finished their undergraduate studies submit the application?", "answer": "Programmes that do not envisage conditional enrolment: Undergraduates who have not yet finished their degree programme, but who plan on passing their degree before the end of the deadline for amendments to pre-register for the requested master's programme, may fill in the pre-registration form and even click on 'Formalise application'. Nevertheless, the application will remain conditional on students finishing their degree programme and paying for their degree certificate to be issued before the end of the amendment deadline for each programme. They will need to attach their official certificate and transcript once the UIB notifies them of the additional required documentation. The deadline to provide this documentation ends after the set period on each master's programme to submit amendments. Please check the 'Deadlines' section on this page. Programmes that envisage conditional enrolment: Undergraduates who have not yet finished their degree programme must fill in the master's pre-registration form and click on 'Formalise application'. If they do not finish their degree studies within the amendment deadline, their application will only be considered where they only have to pass their Final Degree Project and, at most, nine ECTS credits. In any event, applications from graduates will be prioritised."}, {"question": "How many places are available?", "answer": "The number of places that can be made availabe on each master's degree programme is limited, as per that which is set out in the degree verification report, or amendments thereof, provided that it has been approved by ANECA. Each year, the number of general places to be made available on each master's degree programme is proposed and sent to the Academic Commission, the UIB Governing Council, the competent regional department and, where necessary, to the General Conference on University Policy, for approval. Available places: Available places for the 2022-23 academic year and list of programmes with conditional enrolmentDocument reading (https://cep.uib.eu/digitalAssets/675/675432_2022_23-oferta_master_en_revENgb.pdf). Reserved places: Of the total number of places offered in the general call for admissions to master's degree programmes, the following percentages, rounded up to the nearest whole number, shall be reserved: Five per cent of the total number of places for individuals with an accredited disability level equal to or above 33 per cent, as well as for students with permanent educational support needs linked to personal disability circumstances who have required specific resources and support during their previous studies in order to ensure they enjoyed full educational inclusion, Three per cent of the total number of places for high-level and high-performance atheletes. This percentage shall be increased to five per cent on master's programmes linked to sport sciences. Candidates who meet the requirements to apply for admission to more than one category of reserved places may do so. In the event that any reserved places are not taken up, they may be offered as general places. These shall be allocated following the order set out on the waiting list for general places rigurously, and students will have to enrol within the period established by the CEP."}, {"question": "Who can apply for part-time student status?", "answer": "In accordance with article 7 of the UIB Academic Regulations, this is the status that may be requested by students who, for various duly justified reasons, are unable to comply with regulations regarding the minimum number of enrolled credits and academic progress regulations. In this way, the minimum number of credits that students must enrol for is reduced, as well as the number of credits they are required to pass to continue on the programme. First-year full-time students must enrol for at least 30 credits, whereas first-year part-time students are required to enrol for at least 15 credits. Some of the reasons to be granted part-time student status must be accredited on a yearly basis for said status to be renewed, whereas others grant permanent part-time status. In order to apply for part-time student status, students must be in one of the following situations: Working with a minimum dedication of at least half of the maximum duration of the regular working day, Being affected by a physical, sensory or mental disability, to a degree of 33 per cent or above. The disability degree must be accredited by means of a certificate issued by the competent body for it to be validated, Being 45 years of age or above at the start of the academic year, Having family protection status or having to care for dependents, Being a high-level or high-performance athlete, Being enrolled full time during the current academic year on any speciality of the Advanced Diploma in Music, offered by the Advanced Conservatory of Music, Being in any other extraordinary situation that the competent body deems to be decisive in order to be granted part-time student status."}, {"question": "When do you have to apply for part-time student status?", "answer": "Students who wish to take a master's programme part-time, and who fulfil the requirements to do so, must apply for this status at the time of pre-registration, over the same online platform. In the event of a sudden change during the first semester, students may apply for part-time student status outside the set deadline. If they apply before the start of the extended enrolment period, they may reduce the number of credits to the minimum set out in articles 5.8 and 5.9 of the UIB Academic Regulations. In turn, students may only make changes to the number of second-semester credits for which they have enrolled, and in compliance with the academic requirements for subjects in the second semester."}, {"question": "What is the order of precedence?", "answer": "In order to allocate places, applications for admission to a master's degree programme, pathway/specialisation, method and site/centre shall be ranked in order of precedence. If there are n places, places shall be allotted to the first n applicants, and the remainder shall be placed on a waiting list in the same order of precedence. Graduates shall be given priority over students who still need to pass their final degree project and, at most, nine ECTS credits. Selection of the latter will be prioritised as per the following criteria: Specific criteria linked to the entry profile as defined in the degree verification report for each master's programme, Students who only have the foreign language level accreditation pending in order to be awarded the degree, Students whose only pending subject is the final degree project. In the event of a tie, the average mark obtained in the remaining subjects on the programme will be taken into account, Students who have pending credits from other subjects aside from the final degree project. In such instances, priority shall be given to students with fewer pending credits. In the event of a tie, the average mark obtained for all passed credits shall be taken into account. In order to establish the order of precedence for reserved places, these shall be ranked in the corresponding position within each group, depending on the entry requirement fulfilled by the student. Where students qualify for more than one category of reserved places, the following order of precedence will be used: Places reserved for individuals with an accredited disability level equal to or above 33 per cent, Places reserved for high-level/high-performance athletes, General places."}, {"question": "What are amendment lists?", "answer": "The Centre for Postgraduate Studies will review pre-registrations that were correctly formalised. If the pre-registration application contains any form or content error, the Centre for Postgraduate Studies will send an e-mail to students notifying them that they need to amend their pre-registration and stating the deadline for doing so. Moreover, where the master's pre-registration deadline has passed, the Centre for Postgraduate Studies will publish the list of students with amendments to be rectified and the final deadline to do so. Pre-registration applications with amendments still pending after the deadline has passed will not be considered."}, {"question": "How do admission lists work?", "answer": "After verifying the access criteria are fulfilled for the requested master's programme, the Centre for Postgraduate Studies will send the pre-registration application to the corresponding academic commission for said programme for an admission assessment and decision. The admission requirements and criteria for each master's programme are set out in the verification reports for each degree and may be viewed in the 'General Information' tab - 'Admission Criteria' section on the master's factsheet. The Centre for Postgraduate Studies will publish the provisional admissions list, final list and, where applicable, the waiting list, within the set deadlines for each call, in accordance with the resolution issued by the Academic Commission. Admitted students must formalise their enrolment within a set deadline via the Acadèmic app. Failure to do so in the set timeframe will lead to candidates forfeiting their right to enrol or having to re-apply for a place on the master's programme for the same academic year (where they had already been admitted). The admission lists may include students who are on the waiting list. After the enrolment period has finalised, they will be offered any remaining places. Students on the waiting list who are finally admitted will be notified and have to formalise their enrolment within the set deadline. Failure to do so will lead to candidates forfeiting their right to enrol or having to re-apply for a place on the master's programme for the same academic year (where they had already been admitted). Admission lists may include non-admitted students. Those who are not admitted may lodge an appeal, where applicable, to the CEP management team, who will request whatever reports they deem appropriate to issue a decision."}, {"question": "Regulations governing access and admission to master's studies", "answer": "Royal Decree 822/2021 of 28th September that sets out the organisation of university programmes and quality assurance procedureDocument readings: https://www.boe.es/buscar/pdf/2021/BOE-A-2021-15781-consolidado.pdf, Regulatory Agreement 14423/2022 of 23rd March that governs access and admission to official master's programmes: https://seu.uib.cat/fou/acord/14423/, Resolució del vicerector Gestió i Política de Postgrau i Formació Permanent sobre els estudis de màster que no permeten l'accés amb matrícula condicionada l'any acadèmic 2022-23Document reading: https://cep.uib.eu/digitalAssets/678/678186_of_acces-matricula-condicionada_curs-2022_23_v2.pdf"}, {"question": "What grants are available for master's degrees?", "answer": "The Ministry of Education and Vocational Training offers general and mobility grants for university students every year. In addition, other institutions and organisations may occasionally offer grants and financial aid for postgraduate studies. The selection changes every year and, therefore, the UIB has set up a website for current information. You may also check the grants, awards and general financial aid section on the Centre for Postgraduate Studies website."}, {"question": "When is the pre-registration application considered to have been submitted?", "answer": "The pre-registration application is deemed to have been submitted appropriately when students click on 'Formalise application'. Once the application has been formalised, students will receive an e-mail with the application receipt, which, if necessary, will serve as proof that they have submitted an application to study a master's degree programme. Applications which have 'Started' status once the pre-registration deadline as expired shall not be assessed, and shall therefore be withdrawn."}, {"question": "Do I need to attach documentation if my prior studies were taken at the UIB?", "answer": "No, you do not need attach your degree certificate or any official transcript for UIB qualifications."}, {"question": "Do I need to attach documentation if my prior studies were taken at the UIB?", "answer": "No, you do not need attach your degree certificate or any official transcript for UIB qualifications."}, {"question": "What format should the documentation to attach to the application have?", "answer": "Documents that are uploaded to the pre-registration application must be legible and preferably in .pdf format, or be original documents with an electronic signature and a secure verification code."}, {"question": "Where do I have to upload the documents?", "answer": "Each document must be attached to the corresponding place in the application, and must contain all the pages in the correct order."}, {"question": "What validity do scanned documents have?", "answer": "Scanned documents attached to the pre-registration application are accepted provisionally for the application process. Documents with an electronic signature and secure verification code are equivalent to an original document."}, {"question": "When do the originals need to be submitted?", "answer": "In this instance, the originals must be submitted between 1st November and 31st March of the current year. Until original documents are submitted, the CEP is unable to issue any academic certificate for the master's programme, including the final degree certificate. The original and a photocopy must be submitted to the admin services in the Antoni Maria Alcover i Sureda building (Centre for Postgraduate Studies). You must request an appointment to do this step. Where documents are sent by post, a certified copy of each document must be included. The only stamps accepted are those issued by a registered Spanish notary public, a diplomatic service or the issuing body for the document, as long as payment for the certified stamp is accredited."}] 
data = []
for i, entry in enumerate(dataset):
    data.append({
        "question": entry["question"],
        "id": str(i + 1),
        "answers": 
                    {
                        "text": [entry["answer"]],
                        "answer_start": [0]
                    },
        "context": entry["answer"],
        "title": "FAQs of University of the Balearic Islands"
    })
df = pd.DataFrame(data)
df.to_csv("df.csv", index=False) 
df = load_dataset("csv", data_files="df.csv", split='train')
df = df.train_test_split(test_size=0.2)
tokenized_squad = df.map(prepare_train_features, batched=True, remove_columns=df["train"].column_names)
# create the output directory if it doesn't exist
output_dir = '/bloom-1b1-finetuned'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
data_collator = DefaultDataCollator()
training_args = TrainingArguments(

    output_dir=output_dir,

    evaluation_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    num_train_epochs=3,

    weight_decay=0.01 #, push_to_hub=True,

)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_squad["train"],

    eval_dataset=tokenized_squad["test"],
    tokenizer=tokenizer,

    data_collator=data_collator,

)

trainer.train()
# Save the model
model.save_pretrained(output_dir)

